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
void rms_norm_impl(T *out, const T *in, const T *weight, size_t nrow, size_t ncol, float eps) {
    for (size_t i = 0; i < nrow; ++i) {
        float mean = 0.0f;
        for (size_t j = 0; j < ncol; ++j) {
            float x = to_float(in[i * ncol + j]);
            mean += x * x;
        }
        mean /= static_cast<float>(ncol);
        float inv_rms = 1.0f / std::sqrt(mean + eps);

        for (size_t j = 0; j < ncol; ++j) {
            float y = to_float(in[i * ncol + j]) * inv_rms * to_float(weight[j]);
            out[i * ncol + j] = llaisys::utils::cast<T>(y);
        }
    }
}
} // namespace

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() == 2 && out->ndim() == 2, "RMSNorm: in/out must be 2D.");
    CHECK_ARGUMENT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");
    CHECK_ARGUMENT(out->shape() == in->shape(), "RMSNorm: out shape must equal in shape.");
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "RMSNorm: weight shape mismatch.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const size_t nrow = in->shape()[0];
        const size_t ncol = in->shape()[1];
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rms_norm_impl(
                reinterpret_cast<float *>(out->data()),
                reinterpret_cast<const float *>(in->data()),
                reinterpret_cast<const float *>(weight->data()),
                nrow,
                ncol,
                eps);
        case LLAISYS_DTYPE_F16:
            return rms_norm_impl(
                reinterpret_cast<fp16_t *>(out->data()),
                reinterpret_cast<const fp16_t *>(in->data()),
                reinterpret_cast<const fp16_t *>(weight->data()),
                nrow,
                ncol,
                eps);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_impl(
                reinterpret_cast<bf16_t *>(out->data()),
                reinterpret_cast<const bf16_t *>(in->data()),
                reinterpret_cast<const bf16_t *>(weight->data()),
                nrow,
                ncol,
                eps);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rms_norm(out, in, weight, eps);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
