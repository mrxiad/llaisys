from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_float,
    c_int,
    c_int64,
    c_size_t,
    c_void_p,
)

from .llaisys_types import (
    llaisysDataType_t,
    llaisysDeviceType_t,
)


llaisysQwen2Model_t = c_void_p


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelLoadTensor.argtypes = [
        llaisysQwen2Model_t,
        c_char_p,
        c_void_p,
        POINTER(c_size_t),
        c_size_t,
        llaisysDataType_t,
    ]
    lib.llaisysQwen2ModelLoadTensor.restype = c_int

    lib.llaisysQwen2ModelReset.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelReset.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelInferEx.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
        c_int,
        c_float,
        c_float,
    ]
    lib.llaisysQwen2ModelInferEx.restype = c_int64
