#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace {
template <typename T>
float to_float(T value) {
    return llaisys::utils::cast<float>(value);
}

template <typename T>
void self_attention_impl(T *attn_val,
                         const T *q,
                         const T *k,
                         const T *v,
                         size_t qlen,
                         size_t kvlen,
                         size_t nh,
                         size_t nkvh,
                         size_t d,
                         size_t dv,
                         float scale) {
    const size_t group = nh / nkvh;
    std::vector<float> scores(kvlen);
    std::vector<float> probs(kvlen);

    for (size_t t = 0; t < qlen; ++t) {
        const size_t valid_kv = kvlen - qlen + t + 1;
        for (size_t h = 0; h < nh; ++h) {
            const size_t kvh = h / group;
            const size_t q_base = (t * nh + h) * d;

            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t s = 0; s < kvlen; ++s) {
                if (s >= valid_kv) {
                    scores[s] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const size_t k_base = (s * nkvh + kvh) * d;
                float dot = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    dot += to_float(q[q_base + j]) * to_float(k[k_base + j]);
                }
                float score = dot * scale;
                scores[s] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float sum_exp = 0.0f;
            for (size_t s = 0; s < kvlen; ++s) {
                if (s >= valid_kv) {
                    probs[s] = 0.0f;
                } else {
                    float p = std::exp(scores[s] - max_score);
                    probs[s] = p;
                    sum_exp += p;
                }
            }

            const size_t out_base = (t * nh + h) * dv;
            for (size_t j = 0; j < dv; ++j) {
                float value = 0.0f;
                for (size_t s = 0; s < kvlen; ++s) {
                    if (probs[s] == 0.0f) {
                        continue;
                    }
                    const size_t v_base = (s * nkvh + kvh) * dv;
                    value += (probs[s] / sum_exp) * to_float(v[v_base + j]);
                }
                attn_val[out_base + j] = llaisys::utils::cast<T>(value);
            }
        }
    }
}
} // namespace

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3, "SelfAttention: all tensors must be 3D.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "SelfAttention: all tensors must be contiguous.");

    const size_t qlen = q->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t d = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh = k->shape()[1];
    const size_t kd = k->shape()[2];
    const size_t vlen = v->shape()[0];
    const size_t vh = v->shape()[1];
    const size_t dv = v->shape()[2];

    CHECK_ARGUMENT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == dv, "SelfAttention: attn_val shape mismatch.");
    CHECK_ARGUMENT(kvlen == vlen && nkvh == vh, "SelfAttention: k and v shape mismatch.");
    CHECK_ARGUMENT(kd == d, "SelfAttention: q and k head_dim mismatch.");
    CHECK_ARGUMENT(nh % nkvh == 0, "SelfAttention: nhead must be divisible by nkvhead.");
    CHECK_ARGUMENT(kvlen >= qlen, "SelfAttention: kvlen must be >= qlen for causal mask.");
    ASSERT(attn_val->deviceType() == LLAISYS_DEVICE_CPU, "SelfAttention: only CPU is supported now.");

    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl(
            reinterpret_cast<float *>(attn_val->data()),
            reinterpret_cast<const float *>(q->data()),
            reinterpret_cast<const float *>(k->data()),
            reinterpret_cast<const float *>(v->data()),
            qlen,
            kvlen,
            nh,
            nkvh,
            d,
            dv,
            scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl(
            reinterpret_cast<fp16_t *>(attn_val->data()),
            reinterpret_cast<const fp16_t *>(q->data()),
            reinterpret_cast<const fp16_t *>(k->data()),
            reinterpret_cast<const fp16_t *>(v->data()),
            qlen,
            kvlen,
            nh,
            nkvh,
            d,
            dv,
            scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl(
            reinterpret_cast<bf16_t *>(attn_val->data()),
            reinterpret_cast<const bf16_t *>(q->data()),
            reinterpret_cast<const bf16_t *>(k->data()),
            reinterpret_cast<const bf16_t *>(v->data()),
            qlen,
            kvlen,
            nh,
            nkvh,
            d,
            dv,
            scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}
} // namespace llaisys::ops
