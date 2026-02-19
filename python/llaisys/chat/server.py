import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterator, List

from transformers import AutoTokenizer

import llaisys


def _validate_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("messages must be a non-empty list.")
    validated: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object.")
        role = item.get("role")
        content = item.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError("message.role must be one of system/user/assistant.")
        if not isinstance(content, str):
            raise ValueError("message.content must be a string.")
        validated.append({"role": role, "content": content})
    return validated


@dataclass
class ChatRequest:
    model: str
    messages: List[Dict[str, str]]
    stream: bool
    max_tokens: int
    top_k: int
    top_p: float
    temperature: float

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> "ChatRequest":
        model = str(payload.get("model", "qwen2"))
        messages = _validate_messages(payload.get("messages"))
        stream = bool(payload.get("stream", False))
        max_tokens = int(payload.get("max_tokens", 128))
        top_k = int(payload.get("top_k", 50))
        top_p = float(payload.get("top_p", 0.8))
        temperature = float(payload.get("temperature", 0.8))
        return ChatRequest(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )


class ChatService:
    def __init__(self, model_path: str, device_name: str) -> None:
        device = llaisys.DeviceType.CPU if device_name == "cpu" else llaisys.DeviceType.NVIDIA
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = llaisys.models.Qwen2(model_path, device=device)
        self._lock = threading.Lock()

    def _build_prompt_tokens(self, messages: List[Dict[str, str]]) -> List[int]:
        prompt = self._tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self._tokenizer.encode(prompt)

    def complete(self, req: ChatRequest) -> Dict[str, Any]:
        with self._lock:
            prompt_tokens = self._build_prompt_tokens(req.messages)
            out_tokens = self._model.generate(
                prompt_tokens,
                max_new_tokens=req.max_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
            )
            completion_tokens = out_tokens[len(prompt_tokens) :]
            text = self._tokenizer.decode(completion_tokens, skip_special_tokens=True)

        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }

    def complete_stream(self, req: ChatRequest) -> Iterator[Dict[str, Any]]:
        with self._lock:
            prompt_tokens = self._build_prompt_tokens(req.messages)
            created = int(time.time())
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            generated = 0
            for token in self._model.stream_generate(
                prompt_tokens,
                max_new_tokens=req.max_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
            ):
                piece = self._tokenizer.decode([token], skip_special_tokens=True)
                if piece == "":
                    continue
                generated += 1
                yield {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": piece},
                            "finish_reason": None,
                        }
                    ],
                }

            yield {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": generated,
                    "total_tokens": len(prompt_tokens) + generated,
                },
            }


class ChatHandler(BaseHTTPRequestHandler):
    service: ChatService = None  # type: ignore[assignment]

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": {"message": "Not Found"}})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            req = ChatRequest.from_payload(payload)
        except Exception as exc:  # pylint: disable=broad-except
            self._write_json(400, {"error": {"message": str(exc)}})
            return

        if not req.stream:
            try:
                response = self.service.complete(req)
            except Exception as exc:  # pylint: disable=broad-except
                self._write_json(500, {"error": {"message": str(exc)}})
                return
            self._write_json(200, response)
            return

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            for chunk in self.service.complete_stream(req):
                line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                self.wfile.write(line)
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:  # pylint: disable=broad-except
            # Request may be cancelled by client; no extra response needed.
            return

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # Keep console clean for chat interactions.
        _ = (fmt, args)


def run_server(model_path: str, host: str, port: int, device: str) -> None:
    ChatHandler.service = ChatService(model_path=model_path, device_name=device)
    server = HTTPServer((host, port), ChatHandler)
    print(f"Chat server listening on http://{host}:{port}")
    print("Endpoint: POST /v1/chat/completions")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    run_server(model_path=args.model, host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
