from ..libllaisys import DeviceType
from .qwen2 import Qwen2


class Llama(Qwen2):
    """Llama-family model wrapper.

    The current backend path is compatible with the same decoder-only
    transformer tensor layout used by Qwen2/Llama style checkpoints.
    """

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        super().__init__(model_path=model_path, device=device)
