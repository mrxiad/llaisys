#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>

namespace {
template <typename T>
float to_float(T value) {
    return llaisys::utils::cast<float>(value);
}

template <typename T>
void swiglu_impl(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float g = to_float(gate[i]);
        float value = to_float(up[i]) * (g / (1.0f + std::exp(-g)));
        out[i] = llaisys::utils::cast<T>(value);
    }
}
} // namespace

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");
    ASSERT(out->deviceType() == LLAISYS_DEVICE_CPU, "SwiGLU: only CPU is supported now.");

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(gate->data()),
            reinterpret_cast<const float *>(up->data()),
            out->numel());
    case LLAISYS_DTYPE_F16:
        return swiglu_impl(
            reinterpret_cast<fp16_t *>(out->data()),
            reinterpret_cast<const fp16_t *>(gate->data()),
            reinterpret_cast<const fp16_t *>(up->data()),
            out->numel());
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl(
            reinterpret_cast<bf16_t *>(out->data()),
            reinterpret_cast<const bf16_t *>(gate->data()),
            reinterpret_cast<const bf16_t *>(up->data()),
            out->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
