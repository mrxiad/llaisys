#include "op.hpp"

#include "../../utils.hpp"

namespace {
template <typename T>
float to_float(T value) {
    return llaisys::utils::cast<float>(value);
}

template <typename T>
void argmax_impl(int64_t *max_idx, T *max_val, const T *vals, size_t size) {
    size_t best_idx = 0;
    float best_value = to_float(vals[0]);
    for (size_t i = 1; i < size; ++i) {
        float value = to_float(vals[i]);
        if (value > best_value) {
            best_value = value;
            best_idx = i;
        }
    }
    max_idx[0] = static_cast<int64_t>(best_idx);
    max_val[0] = vals[best_idx];
}
} // namespace

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor.");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->shape()[0] == 1, "Argmax: max_idx must be shape [1].");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->shape()[0] == 1, "Argmax: max_val must be shape [1].");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx dtype must be int64.");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous.");
    ASSERT(vals->numel() > 0, "Argmax: vals must not be empty.");
    ASSERT(vals->deviceType() == LLAISYS_DEVICE_CPU, "Argmax: only CPU is supported now.");

    int64_t *max_idx_ptr = reinterpret_cast<int64_t *>(max_idx->data());
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl(
            max_idx_ptr,
            reinterpret_cast<float *>(max_val->data()),
            reinterpret_cast<const float *>(vals->data()),
            vals->numel());
    case LLAISYS_DTYPE_F16:
        return argmax_impl(
            max_idx_ptr,
            reinterpret_cast<fp16_t *>(max_val->data()),
            reinterpret_cast<const fp16_t *>(vals->data()),
            vals->numel());
    case LLAISYS_DTYPE_BF16:
        return argmax_impl(
            max_idx_ptr,
            reinterpret_cast<bf16_t *>(max_val->data()),
            reinterpret_cast<const bf16_t *>(vals->data()),
            vals->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops
