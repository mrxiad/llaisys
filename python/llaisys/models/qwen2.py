import ctypes
import json
from pathlib import Path
from typing import Sequence

import safetensors
import torch

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DataType
from ..libllaisys import DeviceType
from ..libllaisys import LlaisysQwen2Meta
from ..libllaisys import llaisysDataType_t
from ..libllaisys import llaisysDeviceType_t


def _config_dtype_to_llaisys(dtype_name: str) -> DataType:
    dtype_name = dtype_name.lower().replace("torch.", "")
    if dtype_name == "bfloat16":
        return DataType.BF16
    if dtype_name == "float16":
        return DataType.F16
    if dtype_name == "float32":
        return DataType.F32
    raise ValueError(f"Unsupported torch_dtype in config: {dtype_name}")


def _torch_dtype_to_llaisys(dtype: torch.dtype) -> DataType:
    if dtype == torch.bfloat16:
        return DataType.BF16
    if dtype == torch.float16:
        return DataType.F16
    if dtype == torch.float32:
        return DataType.F32
    raise ValueError(f"Unsupported tensor dtype for loading: {dtype}")


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config file: {config_path}")

        # Assignment-3 CPU path: ops and model backend currently run on CPU only.
        if device != DeviceType.CPU:
            raise NotImplementedError("Qwen2 backend currently supports CPU only")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_dtype = _config_dtype_to_llaisys(config.get("torch_dtype", "bfloat16"))

        eos_token_id = config.get("eos_token_id", 0)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        hs = int(config["hidden_size"])
        nh = int(config["num_attention_heads"])
        nkvh = int(config.get("num_key_value_heads", nh))

        meta = LlaisysQwen2Meta(
            dtype=llaisysDataType_t(model_dtype),
            nlayer=int(config["num_hidden_layers"]),
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=hs // nh,
            di=int(config["intermediate_size"]),
            maxseq=int(config["max_position_embeddings"]),
            voc=int(config["vocab_size"]),
            epsilon=float(config.get("rms_norm_eps", 1e-6)),
            theta=float(config.get("rope_theta", 10000.0)),
            end_token=int(eos_token_id),
        )
        self._meta = meta
        self._end_token = int(eos_token_id)

        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            llaisysDeviceType_t(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model backend")

        loaded = 0
        # Load all tensors from safetensors into C++ backend.
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                tensor = data_.get_tensor(name_).contiguous()
                try:
                    dtype = _torch_dtype_to_llaisys(tensor.dtype)
                except ValueError:
                    # Ignore unsupported non-floating tensors not used by runtime.
                    continue

                shape = (ctypes.c_size_t * tensor.dim())(*[int(v) for v in tensor.shape])
                ok = LIB_LLAISYS.llaisysQwen2ModelLoadTensor(
                    self._model,
                    name_.encode("utf-8"),
                    ctypes.c_void_p(int(tensor.data_ptr())),
                    shape,
                    ctypes.c_size_t(tensor.dim()),
                    llaisysDataType_t(dtype),
                )
                if ok == 1:
                    loaded += 1

        if loaded == 0:
            raise RuntimeError("No model tensors were loaded into backend")

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # Assignment #3 checks greedy decoding (top_k=1, top_p=1, temperature=1).
        # Parameters are kept for API compatibility, but backend currently uses argmax.
        _ = (top_k, top_p, temperature)

        tokens = [int(t) for t in inputs]
        if max_new_tokens is None:
            max_new_tokens = 128

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        for _ in range(int(max_new_tokens)):
            if len(tokens) >= int(self._meta.maxseq):
                break
            if len(tokens) == 0:
                break

            token_ids = (ctypes.c_int64 * len(tokens))(*tokens)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, token_ids, ctypes.c_size_t(len(tokens))
                )
            )
            tokens.append(next_token)
            if next_token == self._end_token:
                break

        return tokens
