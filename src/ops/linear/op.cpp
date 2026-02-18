#include "op.hpp"

#include "../../utils.hpp"

namespace {
template <typename T>
float to_float(T value) {
    return llaisys::utils::cast<float>(value);
}

template <typename T>
void linear_impl(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t o = 0; o < n; ++o) {
            float acc = bias == nullptr ? 0.0f : to_float(bias[o]);
            for (size_t j = 0; j < k; ++j) {
                acc += to_float(in[i * k + j]) * to_float(weight[o * k + j]);
            }
            out[i * n + o] = llaisys::utils::cast<T>(acc);
        }
    }
}
} // namespace

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(in->ndim() == 2 && out->ndim() == 2 && weight->ndim() == 2, "Linear: in/out/weight must be 2D.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: in/out/weight must be contiguous.");

    const bool has_bias = bias != nullptr;
    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias must be 1D when provided.");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    const size_t m = in->shape()[0];
    const size_t k = in->shape()[1];
    const size_t n = weight->shape()[0];

    CHECK_ARGUMENT(weight->shape()[1] == k, "Linear: in and weight shapes mismatch.");
    CHECK_ARGUMENT(out->shape()[0] == m && out->shape()[1] == n, "Linear: out shape mismatch.");
    if (has_bias) {
        CHECK_ARGUMENT(bias->shape()[0] == n, "Linear: bias shape mismatch.");
    }

    ASSERT(out->deviceType() == LLAISYS_DEVICE_CPU, "Linear: only CPU is supported now.");

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_impl(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(in->data()),
            reinterpret_cast<const float *>(weight->data()),
            has_bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
            m,
            n,
            k);
    case LLAISYS_DTYPE_F16:
        return linear_impl(
            reinterpret_cast<fp16_t *>(out->data()),
            reinterpret_cast<const fp16_t *>(in->data()),
            reinterpret_cast<const fp16_t *>(weight->data()),
            has_bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr,
            m,
            n,
            k);
    case LLAISYS_DTYPE_BF16:
        return linear_impl(
            reinterpret_cast<bf16_t *>(out->data()),
            reinterpret_cast<const bf16_t *>(in->data()),
            reinterpret_cast<const bf16_t *>(weight->data()),
            has_bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr,
            m,
            n,
            k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
