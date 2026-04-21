import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gemma OpenAI-compatible API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    uvicorn.run("openai_api.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
