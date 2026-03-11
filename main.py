import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from gemma3.model import GEMMA3_CONFIG_270M, build_gemma3_270m
from gemma3.utilities import load_weights_into_gemma


@dataclass
class RunConfig:
    choose_model: str = "270m"
    use_instruct_model: bool = True
    prompt: str = "Give me a short introduction to large language models."
    max_new_tokens: int = 180
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_repo_id(choose_model: str, use_instruct_model: bool) -> str:
    suffix = "-it" if use_instruct_model else ""
    return f"google/gemma-3-{choose_model}{suffix}"


def download_weights(repo_id: str, choose_model: str, local_dir: str) -> Dict[str, torch.Tensor]:
    if choose_model == "270m":
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        return load_file(weights_file)

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = Path(repo_dir) / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weights_dict: Dict[str, torch.Tensor] = {}
    for filename in set(index["weight_map"].values()):
        shard_path = Path(repo_dir) / filename
        shard = load_file(str(shard_path))
        weights_dict.update(shard)
    return weights_dict


class GemmaTokenizer:
    def __init__(self, tokenizer_file_path: str):
        self._tok = Tokenizer.from_file(str(Path(tokenizer_file_path)))
        self.end_of_turn_id = self._tok.encode("<end_of_turn>").ids[-1]
        self.pad_token_id = self.end_of_turn_id
        self.eos_token_id = self.end_of_turn_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def apply_chat_template(user_text: str) -> str:
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


def resolve_tokenizer_path(repo_id: str, local_dir: str) -> str:
    tokenizer_path = Path(local_dir) / "tokenizer.json"
    if tokenizer_path.exists():
        return str(tokenizer_path)

    fallback = Path("tokenizer.json")
    if fallback.exists():
        return str(fallback)

    try:
        return hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
    except Exception as e:
        raise FileNotFoundError(
            "tokenizer.json is unavailable locally and could not be downloaded."
        ) from e


def apply_repetition_penalty_(logits: torch.Tensor, all_token_ids, penalty: Optional[float]) -> torch.Tensor:
    if penalty is None or penalty <= 1.0:
        return logits

    if isinstance(all_token_ids, torch.Tensor):
        token_rows = [torch.unique(all_token_ids[b]).tolist() for b in range(logits.size(0))]
    else:
        token_rows = [list(token_ids) for token_ids in all_token_ids]

    for batch_idx, used_ids in enumerate(token_rows):
        if not used_ids:
            continue
        used_ids_tensor = torch.tensor(used_ids, device=logits.device, dtype=torch.long)
        used_logits = logits[batch_idx, used_ids_tensor]
        logits[batch_idx, used_ids_tensor] = torch.where(
            used_logits < 0,
            used_logits * penalty,
            used_logits / penalty,
        )
    return logits


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)

        to_remove = cdf > top_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text_stream(
    model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Iterator[torch.Tensor]:
    model.eval()
    generated_ids = token_ids.squeeze(0).tolist()
    seen_token_ids = set(generated_ids)
    past_kv = None
    cur_input = token_ids

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out, past_kv = model(cur_input, past_kv=past_kv, use_cache=True)
            logits = out[:, -1, :]
            logits = apply_repetition_penalty_(logits, [seen_token_ids], repetition_penalty)
            next_token = sample_next_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token
            next_id = int(next_token.item())
            generated_ids.append(next_id)
            seen_token_ids.add(next_id)
            cur_input = next_token


def calc_gpu_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024 / 1024:.2f} GB"


def main() -> None:
    run = RunConfig()
    device = get_device()

    model = build_gemma3_270m().to(device)

    repo_id = build_repo_id(run.choose_model, run.use_instruct_model)
    local_dir = Path(repo_id).name

    weights_dict = download_weights(repo_id=repo_id, choose_model=run.choose_model, local_dir=local_dir)
    load_weights_into_gemma(model, GEMMA3_CONFIG_270M, weights_dict)
    del weights_dict

    tokenizer_path = resolve_tokenizer_path(repo_id=repo_id, local_dir=local_dir)
    tokenizer = GemmaTokenizer(tokenizer_path)

    prompt = apply_chat_template(run.prompt) if run.use_instruct_model else run.prompt
    input_token_ids = tokenizer.encode(prompt)
    print(tokenizer.decode(input_token_ids))

    token_ids = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for token in generate_text_stream(
        model=model,
        token_ids=token_ids,
        max_new_tokens=run.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=run.temperature,
        top_p=run.top_p,
        top_k=run.top_k,
        repetition_penalty=run.repetition_penalty,
    ):
        token_id = token.squeeze(0).tolist()
        print(tokenizer.decode(token_id), end="", flush=True)

    if torch.cuda.is_available():
        print(f"\n\nGPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()
