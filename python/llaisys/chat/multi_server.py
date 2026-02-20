import argparse
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

import llaisys


def _validate_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("messages must be a non-empty list.")
    out: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object.")
        role = item.get("role")
        content = item.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError("message.role must be one of system/user/assistant.")
        if not isinstance(content, str):
            raise ValueError("message.content must be a string.")
        out.append({"role": role, "content": content})
    return out


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
        return ChatRequest(
            model=model,
            messages=messages,
            stream=bool(payload.get("stream", False)),
            max_tokens=int(payload.get("max_tokens", 128)),
            top_k=int(payload.get("top_k", 50)),
            top_p=float(payload.get("top_p", 0.8)),
            temperature=float(payload.get("temperature", 0.8)),
        )


@dataclass
class PendingRequest:
    req_id: str
    req: ChatRequest
    created: int
    prompt_tokens: List[int]
    tokens: List[int]
    prompt_len: int
    done_event: threading.Event = field(default_factory=threading.Event)
    chunk_queue: "queue.Queue[Optional[Dict[str, Any]]]" = field(default_factory=queue.Queue)
    error: Optional[str] = None
    final_payload: Optional[Dict[str, Any]] = None

    def completion_tokens(self) -> List[int]:
        return self.tokens[self.prompt_len :]

    def generated_count(self) -> int:
        return len(self.tokens) - self.prompt_len

    def is_done(self, end_token: int) -> bool:
        if self.generated_count() >= self.req.max_tokens:
            return True
        if len(self.tokens) > self.prompt_len and self.tokens[-1] == end_token:
            return True
        return False


class PrefixCachePool:
    def __init__(self, capacity: int = 10000) -> None:
        self._capacity = max(1, capacity)
        self._data: Dict[Tuple[int, ...], int] = {}
        self._order: Deque[Tuple[int, ...]] = deque()
        self._lock = threading.Lock()

    def get(self, prefix: List[int]) -> Optional[int]:
        key = tuple(prefix)
        with self._lock:
            return self._data.get(key)

    def put(self, prefix: List[int], next_token: int) -> None:
        key = tuple(prefix)
        with self._lock:
            if key not in self._data:
                self._order.append(key)
            self._data[key] = int(next_token)
            while len(self._order) > self._capacity:
                old = self._order.popleft()
                self._data.pop(old, None)


class MultiUserScheduler:
    def __init__(self, model_path: str, device_name: str, batch_size: int = 4, prefix_cache_cap: int = 50000) -> None:
        device = llaisys.DeviceType.CPU if device_name == "cpu" else llaisys.DeviceType.NVIDIA
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = llaisys.models.load_model(model_path, device=device)
        self._end_token = self._model._end_token  # noqa: SLF001

        self._batch_size = max(1, int(batch_size))
        self._prefix_cache = PrefixCachePool(prefix_cache_cap)

        self._incoming: "queue.Queue[PendingRequest]" = queue.Queue()
        self._pending: Deque[PendingRequest] = deque()
        self._shutdown = threading.Event()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._shutdown.set()
        self._worker.join(timeout=5)

    def submit(self, req: ChatRequest) -> PendingRequest:
        prompt = self._tokenizer.apply_chat_template(
            conversation=req.messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_tokens = self._tokenizer.encode(prompt)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        item = PendingRequest(
            req_id=req_id,
            req=req,
            created=int(time.time()),
            prompt_tokens=prompt_tokens,
            tokens=prompt_tokens.copy(),
            prompt_len=len(prompt_tokens),
        )
        self._incoming.put(item)
        return item

    def _run_loop(self) -> None:
        while not self._shutdown.is_set():
            self._drain_incoming()
            if not self._pending:
                time.sleep(0.002)
                continue

            batch: List[PendingRequest] = []
            while self._pending and len(batch) < self._batch_size:
                batch.append(self._pending.popleft())

            for item in batch:
                if item.done_event.is_set():
                    continue
                try:
                    self._infer_one_round(item)
                except Exception as exc:  # pylint: disable=broad-except
                    item.error = str(exc)
                    item.done_event.set()
                    if item.req.stream:
                        item.chunk_queue.put(
                            {
                                "error": {
                                    "message": item.error,
                                }
                            }
                        )
                        item.chunk_queue.put(None)
                    continue

                if item.is_done(self._end_token):
                    self._finalize(item)
                else:
                    self._pending.append(item)

    def _drain_incoming(self) -> None:
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                return
            self._pending.append(item)

    def _infer_one_round(self, item: PendingRequest) -> None:
        # Prefix cache reuse is safe only for deterministic decode.
        deterministic = (
            item.req.top_k == 1
            and item.req.top_p >= 1.0
            and abs(item.req.temperature - 1.0) < 1e-6
        )
        next_token = None
        if deterministic:
            next_token = self._prefix_cache.get(item.tokens)

        if next_token is None:
            out = self._model.generate(
                item.tokens,
                max_new_tokens=1,
                top_k=item.req.top_k,
                top_p=item.req.top_p,
                temperature=item.req.temperature,
            )
            if len(out) <= len(item.tokens):
                next_token = self._end_token
            else:
                next_token = int(out[-1])
            if deterministic:
                self._prefix_cache.put(item.tokens, next_token)

        item.tokens.append(int(next_token))

        if item.req.stream:
            piece = self._tokenizer.decode([next_token], skip_special_tokens=True)
            if piece != "":
                item.chunk_queue.put(
                    {
                        "id": item.req_id,
                        "object": "chat.completion.chunk",
                        "created": item.created,
                        "model": item.req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                )

    def _finalize(self, item: PendingRequest) -> None:
        completion_tokens = item.completion_tokens()
        text = self._tokenizer.decode(completion_tokens, skip_special_tokens=True)
        item.final_payload = {
            "id": item.req_id,
            "object": "chat.completion",
            "created": item.created,
            "model": item.req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": item.prompt_len,
                "completion_tokens": len(completion_tokens),
                "total_tokens": item.prompt_len + len(completion_tokens),
            },
        }

        if item.req.stream:
            item.chunk_queue.put(
                {
                    "id": item.req_id,
                    "object": "chat.completion.chunk",
                    "created": item.created,
                    "model": item.req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": item.final_payload["usage"],
                }
            )
            item.chunk_queue.put(None)
        item.done_event.set()


class MultiChatHandler(BaseHTTPRequestHandler):
    scheduler: MultiUserScheduler = None  # type: ignore[assignment]

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
            item = self.scheduler.submit(req)
        except Exception as exc:  # pylint: disable=broad-except
            self._write_json(400, {"error": {"message": str(exc)}})
            return

        if not req.stream:
            item.done_event.wait()
            if item.error is not None:
                self._write_json(500, {"error": {"message": item.error}})
            else:
                self._write_json(200, item.final_payload if item.final_payload else {"error": {"message": "empty response"}})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        while True:
            chunk = item.chunk_queue.get()
            if chunk is None:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(line)
            self.wfile.flush()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        _ = (fmt, args)


def run_server(model_path: str, host: str, port: int, device: str, batch_size: int) -> None:
    MultiChatHandler.scheduler = MultiUserScheduler(
        model_path=model_path,
        device_name=device,
        batch_size=batch_size,
    )
    server = ThreadingHTTPServer((host, port), MultiChatHandler)
    print(f"Multi-user chat server listening on http://{host}:{port}")
    print("Endpoint: POST /v1/chat/completions")
    print(f"Scheduler batch_size={batch_size}")
    try:
        server.serve_forever()
    finally:
        MultiChatHandler.scheduler.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()
    run_server(
        model_path=args.model,
        host=args.host,
        port=args.port,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
