#include "op.hpp"

#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(out->isContiguous() && in->isContiguous(), "Rearrange: only contiguous tensors are supported now.");

    std::memcpy(out->data(), in->data(), out->numel() * out->elementSize());
}
} // namespace llaisys::ops
