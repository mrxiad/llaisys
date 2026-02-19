#include "op.hpp"

#include "../../utils.hpp"
#include "../nvidia/op_nvidia.hpp"

#include <cstring>

namespace {
template <typename T>
void embedding_impl(T *out, const int64_t *index, const T *weight, size_t nindex, size_t nembed) {
    for (size_t i = 0; i < nindex; ++i) {
        std::memcpy(out + i * nembed, weight + static_cast<size_t>(index[i]) * nembed, nembed * sizeof(T));
    }
}
} // namespace

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be 1D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: out must be 2D.");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index dtype must be int64.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "Embedding: out rows must equal index length.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "Embedding: out columns must equal weight embedding size.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index->data());
        const size_t nindex = index->shape()[0];
        const size_t nrow = weight->shape()[0];
        const size_t nembed = weight->shape()[1];
        for (size_t i = 0; i < nindex; ++i) {
            CHECK_ARGUMENT(idx_ptr[i] >= 0 && static_cast<size_t>(idx_ptr[i]) < nrow, "Embedding: index out of range.");
        }

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return embedding_impl(
                reinterpret_cast<float *>(out->data()),
                idx_ptr,
                reinterpret_cast<const float *>(weight->data()),
                nindex,
                nembed);
        case LLAISYS_DTYPE_F16:
            return embedding_impl(
                reinterpret_cast<fp16_t *>(out->data()),
                idx_ptr,
                reinterpret_cast<const fp16_t *>(weight->data()),
                nindex,
                nembed);
        case LLAISYS_DTYPE_BF16:
            return embedding_impl(
                reinterpret_cast<bf16_t *>(out->data()),
                idx_ptr,
                reinterpret_cast<const bf16_t *>(weight->data()),
                nindex,
                nembed);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (weight->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::embedding(out, index, weight);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
