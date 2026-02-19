#include "op.hpp"

#include "../../utils.hpp"
#include "../nvidia/op_nvidia.hpp"

#include <algorithm>
#include <thread>
#include <vector>

namespace {
template <typename T>
float to_float(T value) {
    return llaisys::utils::cast<float>(value);
}

size_t recommended_threads(size_t m, size_t n, size_t k) {
    const unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads <= 1 || m < 2) {
        return 1;
    }
    const double work = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    if (work < 4.0e6) {
        return 1;
    }
    return std::min(static_cast<size_t>(hw_threads), m);
}

template <typename Fn>
void parallel_for_rows(size_t m, size_t num_threads, Fn &&fn) {
    if (num_threads <= 1 || m <= 1) {
        fn(0, m);
        return;
    }

    const size_t chunk = (m + num_threads - 1) / num_threads;
    std::vector<std::thread> workers;
    workers.reserve(num_threads - 1);

    size_t begin = 0;
    for (size_t t = 0; t + 1 < num_threads && begin < m; ++t) {
        const size_t end = std::min(m, begin + chunk);
        workers.emplace_back([begin, end, &fn]() { fn(begin, end); });
        begin = end;
    }
    if (begin < m) {
        fn(begin, m);
    }

    for (auto &worker : workers) {
        worker.join();
    }
}

void linear_rows_f32(
    float *out, const float *in, const float *weight, const float *bias, size_t row_begin, size_t row_end, size_t n, size_t k) {
    for (size_t i = row_begin; i < row_end; ++i) {
        const float *in_row = in + i * k;
        float *out_row = out + i * n;
        for (size_t o = 0; o < n; ++o) {
            const float *w_row = weight + o * k;
            float acc = bias == nullptr ? 0.0f : bias[o];
            size_t j = 0;
            for (; j + 7 < k; j += 8) {
                acc += in_row[j + 0] * w_row[j + 0];
                acc += in_row[j + 1] * w_row[j + 1];
                acc += in_row[j + 2] * w_row[j + 2];
                acc += in_row[j + 3] * w_row[j + 3];
                acc += in_row[j + 4] * w_row[j + 4];
                acc += in_row[j + 5] * w_row[j + 5];
                acc += in_row[j + 6] * w_row[j + 6];
                acc += in_row[j + 7] * w_row[j + 7];
            }
            for (; j < k; ++j) {
                acc += in_row[j] * w_row[j];
            }
            out_row[o] = acc;
        }
    }
}

void linear_impl_f32(float *out, const float *in, const float *weight, const float *bias, size_t m, size_t n, size_t k) {
    const size_t num_threads = recommended_threads(m, n, k);
    parallel_for_rows(m, num_threads, [&](size_t begin, size_t end) {
        linear_rows_f32(out, in, weight, bias, begin, end, n, k);
    });
}

template <typename T>
void linear_impl_generic(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    const size_t num_threads = recommended_threads(m, n, k);
    parallel_for_rows(m, num_threads, [&](size_t row_begin, size_t row_end) {
        for (size_t i = row_begin; i < row_end; ++i) {
            const T *in_row = in + i * k;
            T *out_row = out + i * n;
            for (size_t o = 0; o < n; ++o) {
                const T *w_row = weight + o * k;
                float acc = bias == nullptr ? 0.0f : to_float(bias[o]);
                for (size_t j = 0; j < k; ++j) {
                    acc += to_float(in_row[j]) * to_float(w_row[j]);
                }
                out_row[o] = llaisys::utils::cast<T>(acc);
            }
        }
    });
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

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return linear_impl_f32(
                reinterpret_cast<float *>(out->data()),
                reinterpret_cast<const float *>(in->data()),
                reinterpret_cast<const float *>(weight->data()),
                has_bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
                m,
                n,
                k);
        case LLAISYS_DTYPE_F16:
            return linear_impl_generic(
                reinterpret_cast<fp16_t *>(out->data()),
                reinterpret_cast<const fp16_t *>(in->data()),
                reinterpret_cast<const fp16_t *>(weight->data()),
                has_bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr,
                m,
                n,
                k);
        case LLAISYS_DTYPE_BF16:
            return linear_impl_generic(
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

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::linear(out, in, weight, bias);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
