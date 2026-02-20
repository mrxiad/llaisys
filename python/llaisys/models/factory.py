import json
from pathlib import Path

from ..libllaisys import DeviceType
from .llama import Llama
from .qwen2 import Qwen2


def _read_config(model_path: str | Path) -> dict:
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def detect_model_type(model_path: str | Path) -> str:
    config = _read_config(model_path)
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(item).lower() for item in config.get("architectures", [])]

    if model_type == "llama" or any("llama" in item for item in architectures):
        return "llama"
    if model_type == "qwen2" or any("qwen2" in item for item in architectures):
        return "qwen2"

    # Fallback: both families share the same required decoder config fields.
    required = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "vocab_size",
    )
    if all(key in config for key in required):
        return "qwen2"

    raise ValueError(
        "Unsupported model type. Expected llama/qwen2 style config with decoder-only fields."
    )


def load_model(model_path: str | Path, device: DeviceType = DeviceType.CPU):
    model_type = detect_model_type(model_path)
    if model_type == "llama":
        return Llama(model_path, device=device)
    return Qwen2(model_path, device=device)
