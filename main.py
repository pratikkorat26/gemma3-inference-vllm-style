from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from engine.runtime import GemmaRuntime, apply_chat_template, get_device
from engine.sampling import apply_repetition_penalty_, sample_next_token


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

    runtime = GemmaRuntime(
        choose_model=run.choose_model,
        use_instruct_model=run.use_instruct_model,
        device=device,
    )

    prompt = apply_chat_template(run.prompt) if run.use_instruct_model else run.prompt
    input_token_ids = runtime.tokenizer.encode(prompt)
    print(runtime.tokenizer.decode(input_token_ids))

    token_ids = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for token in generate_text_stream(
        model=runtime.model,
        token_ids=token_ids,
        max_new_tokens=run.max_new_tokens,
        eos_token_id=runtime.tokenizer.eos_token_id,
        temperature=run.temperature,
        top_p=run.top_p,
        top_k=run.top_k,
        repetition_penalty=run.repetition_penalty,
    ):
        token_id = int(token.item())
        print(runtime.tokenizer.decode([token_id]), end="", flush=True)

    if torch.cuda.is_available():
        print(f"\n\nGPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()