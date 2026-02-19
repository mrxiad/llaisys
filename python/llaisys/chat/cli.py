import argparse
import json
import urllib.request
from typing import Any, Dict, List


def _post_json(url: str, payload: Dict[str, Any]):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=600)  # noqa: S310


def _chat_once(url: str, payload: Dict[str, Any]) -> str:
    if not payload.get("stream", False):
        with _post_json(url, payload) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    out = ""
    with _post_json(url, payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                print(piece, end="", flush=True)
                out += piece
    print("")
    return out


def run_cli(base_url: str, model: str, stream: bool, max_tokens: int, top_k: int, top_p: float, temperature: float) -> None:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    history: List[Dict[str, str]] = []

    print("Chat CLI ready. Commands: /reset /exit")
    while True:
        user_text = input("You> ").strip()
        if user_text == "":
            continue
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/reset":
            history.clear()
            print("Conversation cleared.")
            continue

        history.append({"role": "user", "content": user_text})
        payload = {
            "model": model,
            "messages": history,
            "stream": stream,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
        }

        print("Assistant> ", end="", flush=True)
        assistant = _chat_once(endpoint, payload)
        history.append({"role": "assistant", "content": assistant})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", type=str)
    parser.add_argument("--model", default="qwen2", type=str)
    parser.add_argument("--max-tokens", default=256, type=int)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--top-p", default=0.8, type=float)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()
    run_cli(
        base_url=args.base_url,
        model=args.model,
        stream=not args.no_stream,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
