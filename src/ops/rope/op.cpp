#include "op.hpp"

#include "../../utils.hpp"
#include "../nvidia/op_nvidia.hpp"

#include <cmath>

namespace {
template <typename T>
float to_float(T value) {
    return llaisys::utils::cast<float>(value);
}

template <typename T>
void rope_impl(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t d, float theta) {
    const size_t half = d / 2;
    for (size_t i = 0; i < seqlen; ++i) {
        const float pos = static_cast<float>(pos_ids[i]);
        for (size_t h = 0; h < nhead; ++h) {
            const size_t base = (i * nhead + h) * d;
            for (size_t j = 0; j < half; ++j) {
                const float freq = pos / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
                const float c = std::cos(freq);
                const float s = std::sin(freq);
                const float a = to_float(in[base + j]);
                const float b = to_float(in[base + half + j]);
                const float out_a = a * c - b * s;
                const float out_b = b * c + a * s;
                out[base + j] = llaisys::utils::cast<T>(out_a);
                out[base + half + j] = llaisys::utils::cast<T>(out_b);
            }
        }
    }
}
} // namespace

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(in->ndim() == 3 && out->ndim() == 3, "RoPE: in/out must be 3D.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids dtype must be int64.");
    CHECK_ARGUMENT(out->shape() == in->shape(), "RoPE: out shape mismatch.");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids length mismatch.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "RoPE: head dimension must be even.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const size_t seqlen = in->shape()[0];
        const size_t nhead = in->shape()[1];
        const size_t d = in->shape()[2];

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rope_impl(
                reinterpret_cast<float *>(out->data()),
                reinterpret_cast<const float *>(in->data()),
                reinterpret_cast<const int64_t *>(pos_ids->data()),
                seqlen,
                nhead,
                d,
                theta);
        case LLAISYS_DTYPE_F16:
            return rope_impl(
                reinterpret_cast<fp16_t *>(out->data()),
                reinterpret_cast<const fp16_t *>(in->data()),
                reinterpret_cast<const int64_t *>(pos_ids->data()),
                seqlen,
                nhead,
                d,
                theta);
        case LLAISYS_DTYPE_BF16:
            return rope_impl(
                reinterpret_cast<bf16_t *>(out->data()),
                reinterpret_cast<const bf16_t *>(in->data()),
                reinterpret_cast<const int64_t *>(pos_ids->data()),
                seqlen,
                nhead,
                d,
                theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rope(out, in, pos_ids, theta);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
