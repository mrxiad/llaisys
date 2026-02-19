#include "op.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace {
template <typename T>
float to_float(T value) {
    return llaisys::utils::cast<float>(value);
}

template <typename T>
void load_logits(std::vector<float> &dst, const T *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = to_float(src[i]);
    }
}

thread_local std::mt19937 g_rng{std::random_device{}()};
} // namespace

namespace llaisys::ops {
void sample(tensor_t sampled_idx, tensor_t logits, float temperature, int top_k, float top_p) {
    CHECK_SAME_DEVICE(sampled_idx, logits);
    CHECK_ARGUMENT(logits->ndim() == 1, "Sample: logits must be a 1D tensor.");
    CHECK_ARGUMENT(sampled_idx->ndim() == 1 && sampled_idx->shape()[0] == 1, "Sample: sampled_idx must be shape [1].");
    CHECK_ARGUMENT(sampled_idx->dtype() == LLAISYS_DTYPE_I64, "Sample: sampled_idx dtype must be int64.");
    CHECK_ARGUMENT(
        logits->dtype() == LLAISYS_DTYPE_F32 || logits->dtype() == LLAISYS_DTYPE_F16 || logits->dtype() == LLAISYS_DTYPE_BF16,
        "Sample: logits dtype must be f32/f16/bf16.");
    CHECK_ARGUMENT(temperature > 0.0f, "Sample: temperature must be > 0.");
    CHECK_ARGUMENT(top_p > 0.0f && top_p <= 1.0f, "Sample: top_p must be in (0, 1].");
    ASSERT(sampled_idx->isContiguous() && logits->isContiguous(), "Sample: tensors must be contiguous.");
    ASSERT(logits->numel() > 0, "Sample: logits must not be empty.");

    CHECK_ARGUMENT(logits->deviceType() == LLAISYS_DEVICE_CPU, "Sample: currently only CPU device is supported.");

    const size_t n = logits->numel();
    std::vector<float> scores(n);
    switch (logits->dtype()) {
    case LLAISYS_DTYPE_F32:
        load_logits(scores, reinterpret_cast<const float *>(logits->data()), n);
        break;
    case LLAISYS_DTYPE_F16:
        load_logits(scores, reinterpret_cast<const fp16_t *>(logits->data()), n);
        break;
    case LLAISYS_DTYPE_BF16:
        load_logits(scores, reinterpret_cast<const bf16_t *>(logits->data()), n);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(logits->dtype());
    }

    if (std::abs(temperature - 1.0f) > 1e-6f) {
        const float inv_temperature = 1.0f / temperature;
        for (float &v : scores) {
            v *= inv_temperature;
        }
    }

    size_t k = n;
    if (top_k > 0) {
        k = std::min(n, static_cast<size_t>(top_k));
    }

    std::vector<int64_t> candidates(n);
    std::iota(candidates.begin(), candidates.end(), 0);
    auto score_greater = [&](int64_t lhs, int64_t rhs) { return scores[static_cast<size_t>(lhs)] > scores[static_cast<size_t>(rhs)]; };

    if (k < n) {
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), score_greater);
        candidates.resize(k);
    } else {
        std::sort(candidates.begin(), candidates.end(), score_greater);
    }

    std::vector<double> weights(candidates.size(), 0.0);
    if (top_p < 1.0f) {
        const float max_score = scores[static_cast<size_t>(candidates[0])];
        double sum_exp = 0.0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const double w = std::exp(static_cast<double>(scores[static_cast<size_t>(candidates[i])] - max_score));
            weights[i] = w;
            sum_exp += w;
        }

        double cumulative = 0.0;
        size_t keep = 0;
        for (; keep < candidates.size(); ++keep) {
            cumulative += weights[keep] / sum_exp;
            if (cumulative >= static_cast<double>(top_p)) {
                ++keep;
                break;
            }
        }
        if (keep == 0) {
            keep = 1;
        }
        candidates.resize(keep);
        weights.resize(keep);
    }

    if (weights.empty()) {
        const float max_score = scores[static_cast<size_t>(candidates[0])];
        weights.resize(candidates.size(), 0.0);
        for (size_t i = 0; i < candidates.size(); ++i) {
            weights[i] = std::exp(static_cast<double>(scores[static_cast<size_t>(candidates[i])] - max_score));
        }
    }

    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    const size_t pick = dist(g_rng);
    reinterpret_cast<int64_t *>(sampled_idx->data())[0] = candidates[pick];
}
} // namespace llaisys::ops
