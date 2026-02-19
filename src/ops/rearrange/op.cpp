#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(out->isContiguous() && in->isContiguous(), "Rearrange: only contiguous tensors are supported now.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(out->data(), in->data(), out->numel() * out->elementSize());
        return;
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    core::context().runtime().api()->memcpy_sync(
        out->data(),
        in->data(),
        out->numel() * out->elementSize(),
        LLAISYS_MEMCPY_D2D);
}
} // namespace llaisys::ops
