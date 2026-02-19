#include "llaisys/models/qwen2.h"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/sample/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {
using llaisys::Tensor;
using llaisys::bf16_t;
using llaisys::fp16_t;
using llaisys::tensor_t;

struct Qwen2WeightsNative {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    Qwen2WeightsNative w{};
    // Per-layer KV cache, shape: [kv_capacity, nkvh, dh].
    std::vector<tensor_t> k_cache{};
    std::vector<tensor_t> v_cache{};
    size_t kv_capacity = 0;

    // Prefix cache bookkeeping:
    // cached_tokens token ids in cached_token_ids are already materialized in KV cache.
    size_t cached_tokens = 0;
    std::vector<int64_t> cached_token_ids{};

    // Memoized greedy decode result for the last seen ntoken.
    bool cached_next_token_valid = false;
    size_t cached_next_for_ntoken = 0;
    int64_t cached_next_token = -1;
};

namespace {
std::vector<size_t> to_shape_vec(size_t *shape, size_t ndim) {
    CHECK_ARGUMENT(ndim > 0, "Qwen2: tensor ndim must be > 0");
    CHECK_ARGUMENT(shape != nullptr, "Qwen2: tensor shape pointer is null");
    return std::vector<size_t>(shape, shape + ndim);
}

bool is_supported_float_dtype(llaisysDataType_t dtype) {
    return dtype == LLAISYS_DTYPE_F32 || dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16;
}

void ensure_model_device_supported(const LlaisysQwen2Model *model) {
    CHECK_ARGUMENT(model->device == LLAISYS_DEVICE_CPU, "Qwen2: only CPU device is supported currently.");
}

tensor_t create_model_tensor(const LlaisysQwen2Model *model, const std::vector<size_t> &shape, llaisysDataType_t dtype) {
    return Tensor::create(shape, dtype, model->device, model->device_id);
}

template <typename DstT, typename SrcT>
void cast_copy(std::vector<DstT> &dst, const SrcT *src) {
    const size_t n = dst.size();
    for (size_t i = 0; i < n; ++i) {
        float value = llaisys::utils::cast<float>(src[i]);
        dst[i] = llaisys::utils::cast<DstT>(value);
    }
}

template <typename DstT>
void load_converted_impl(tensor_t dst, const void *src, llaisysDataType_t src_dtype) {
    const size_t n = dst->numel();
    std::vector<DstT> tmp(n);

    switch (src_dtype) {
    case LLAISYS_DTYPE_F32:
        cast_copy(tmp, reinterpret_cast<const float *>(src));
        break;
    case LLAISYS_DTYPE_F16:
        cast_copy(tmp, reinterpret_cast<const fp16_t *>(src));
        break;
    case LLAISYS_DTYPE_BF16:
        cast_copy(tmp, reinterpret_cast<const bf16_t *>(src));
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(src_dtype);
    }

    dst->load(tmp.data());
}

void load_tensor_casted(tensor_t dst, const void *src, llaisysDataType_t src_dtype) {
    CHECK_ARGUMENT(dst != nullptr, "Qwen2: destination tensor must not be null.");
    CHECK_ARGUMENT(src != nullptr, "Qwen2: source pointer must not be null.");
    CHECK_ARGUMENT(is_supported_float_dtype(dst->dtype()), "Qwen2: unsupported destination weight dtype.");
    CHECK_ARGUMENT(is_supported_float_dtype(src_dtype), "Qwen2: unsupported source weight dtype.");

    if (src_dtype == dst->dtype()) {
        dst->load(src);
        return;
    }

    CHECK_ARGUMENT(dst->deviceType() == LLAISYS_DEVICE_CPU, "Qwen2: casted load currently supports CPU tensors only.");
    switch (dst->dtype()) {
    case LLAISYS_DTYPE_F32:
        return load_converted_impl<float>(dst, src, src_dtype);
    case LLAISYS_DTYPE_F16:
        return load_converted_impl<fp16_t>(dst, src, src_dtype);
    case LLAISYS_DTYPE_BF16:
        return load_converted_impl<bf16_t>(dst, src, src_dtype);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dst->dtype());
    }
}

bool parse_layer_name(const std::string &name, size_t &layer, std::string &suffix) {
    static const std::string prefix = "model.layers.";
    if (name.rfind(prefix, 0) != 0) {
        return false;
    }

    size_t layer_start = prefix.size();
    size_t dot_pos = name.find('.', layer_start);
    if (dot_pos == std::string::npos || dot_pos == layer_start) {
        return false;
    }

    size_t value = 0;
    for (size_t i = layer_start; i < dot_pos; ++i) {
        char c = name[i];
        if (c < '0' || c > '9') {
            return false;
        }
        value = value * 10 + static_cast<size_t>(c - '0');
    }

    layer = value;
    suffix = name.substr(dot_pos + 1);
    return true;
}

void reset_generation_state(LlaisysQwen2Model *model) {
    model->cached_tokens = 0;
    model->cached_token_ids.clear();
    model->cached_next_token_valid = false;
    model->cached_next_for_ntoken = 0;
    model->cached_next_token = -1;
}

void check_model_ready(const LlaisysQwen2Model *model) {
    CHECK_ARGUMENT(model->w.in_embed != nullptr, "Qwen2: missing model.embed_tokens.weight");
    CHECK_ARGUMENT(model->w.out_embed != nullptr, "Qwen2: missing lm_head.weight");
    CHECK_ARGUMENT(model->w.out_norm_w != nullptr, "Qwen2: missing model.norm.weight");

    const size_t nlayer = model->meta.nlayer;
    for (size_t i = 0; i < nlayer; ++i) {
        CHECK_ARGUMENT(model->w.attn_norm_w[i] != nullptr, "Qwen2: missing input_layernorm.weight");
        CHECK_ARGUMENT(model->w.attn_q_w[i] != nullptr, "Qwen2: missing self_attn.q_proj.weight");
        CHECK_ARGUMENT(model->w.attn_k_w[i] != nullptr, "Qwen2: missing self_attn.k_proj.weight");
        CHECK_ARGUMENT(model->w.attn_v_w[i] != nullptr, "Qwen2: missing self_attn.v_proj.weight");
        CHECK_ARGUMENT(model->w.attn_o_w[i] != nullptr, "Qwen2: missing self_attn.o_proj.weight");
        CHECK_ARGUMENT(model->w.mlp_norm_w[i] != nullptr, "Qwen2: missing post_attention_layernorm.weight");
        CHECK_ARGUMENT(model->w.mlp_gate_w[i] != nullptr, "Qwen2: missing mlp.gate_proj.weight");
        CHECK_ARGUMENT(model->w.mlp_up_w[i] != nullptr, "Qwen2: missing mlp.up_proj.weight");
        CHECK_ARGUMENT(model->w.mlp_down_w[i] != nullptr, "Qwen2: missing mlp.down_proj.weight");
    }
}

void ensure_kv_capacity(LlaisysQwen2Model *model, size_t required_len) {
    CHECK_ARGUMENT(required_len <= model->meta.maxseq, "Qwen2: input length exceeds maxseq.");

    if (required_len <= model->kv_capacity) {
        return;
    }

    size_t new_capacity = model->kv_capacity == 0 ? size_t(1) : model->kv_capacity;
    while (new_capacity < required_len && new_capacity < model->meta.maxseq) {
        new_capacity *= 2;
    }
    if (new_capacity > model->meta.maxseq) {
        new_capacity = model->meta.maxseq;
    }
    CHECK_ARGUMENT(new_capacity >= required_len, "Qwen2: failed to expand KV cache.");

    // Grow cache with data migration so existing prefix can be reused.
    const size_t nlayer = model->meta.nlayer;
    for (size_t l = 0; l < nlayer; ++l) {
        tensor_t new_k = create_model_tensor(
            model,
            {new_capacity, model->meta.nkvh, model->meta.dh},
            model->meta.dtype);
        tensor_t new_v = create_model_tensor(
            model,
            {new_capacity, model->meta.nkvh, model->meta.dh},
            model->meta.dtype);

        if (model->cached_tokens > 0 && model->k_cache[l] != nullptr && model->v_cache[l] != nullptr) {
            tensor_t old_k_prefix = model->k_cache[l]->slice(0, 0, model->cached_tokens);
            tensor_t old_v_prefix = model->v_cache[l]->slice(0, 0, model->cached_tokens);
            tensor_t new_k_prefix = new_k->slice(0, 0, model->cached_tokens);
            tensor_t new_v_prefix = new_v->slice(0, 0, model->cached_tokens);
            llaisys::ops::rearrange(new_k_prefix, old_k_prefix);
            llaisys::ops::rearrange(new_v_prefix, old_v_prefix);
        }

        model->k_cache[l] = std::move(new_k);
        model->v_cache[l] = std::move(new_v);
    }
    model->kv_capacity = new_capacity;
}

bool set_named_weight(
    LlaisysQwen2Model *model,
    const std::string &name,
    const void *data,
    const std::vector<size_t> &shape,
    llaisysDataType_t src_dtype) {
    const size_t hs = model->meta.hs;
    const size_t nh = model->meta.nh;
    const size_t nkvh = model->meta.nkvh;
    const size_t dh = model->meta.dh;
    const size_t di = model->meta.di;
    const size_t voc = model->meta.voc;

    auto make_weight = [&](const std::vector<size_t> &expect_shape) -> tensor_t {
        CHECK_ARGUMENT(shape == expect_shape, ("Qwen2: shape mismatch for " + name).c_str());
        tensor_t t = create_model_tensor(model, shape, model->meta.dtype);
        load_tensor_casted(t, data, src_dtype);
        return t;
    };

    if (name == "model.embed_tokens.weight") {
        model->w.in_embed = make_weight({voc, hs});
        return true;
    }
    if (name == "lm_head.weight") {
        model->w.out_embed = make_weight({voc, hs});
        return true;
    }
    if (name == "model.norm.weight") {
        model->w.out_norm_w = make_weight({hs});
        return true;
    }

    size_t layer = 0;
    std::string suffix;
    if (!parse_layer_name(name, layer, suffix)) {
        return false;
    }
    if (layer >= model->meta.nlayer) {
        return false;
    }

    if (suffix == "input_layernorm.weight") {
        model->w.attn_norm_w[layer] = make_weight({hs});
        return true;
    }
    if (suffix == "self_attn.q_proj.weight") {
        model->w.attn_q_w[layer] = make_weight({nh * dh, hs});
        return true;
    }
    if (suffix == "self_attn.q_proj.bias") {
        model->w.attn_q_b[layer] = make_weight({nh * dh});
        return true;
    }
    if (suffix == "self_attn.k_proj.weight") {
        model->w.attn_k_w[layer] = make_weight({nkvh * dh, hs});
        return true;
    }
    if (suffix == "self_attn.k_proj.bias") {
        model->w.attn_k_b[layer] = make_weight({nkvh * dh});
        return true;
    }
    if (suffix == "self_attn.v_proj.weight") {
        model->w.attn_v_w[layer] = make_weight({nkvh * dh, hs});
        return true;
    }
    if (suffix == "self_attn.v_proj.bias") {
        model->w.attn_v_b[layer] = make_weight({nkvh * dh});
        return true;
    }
    if (suffix == "self_attn.o_proj.weight") {
        model->w.attn_o_w[layer] = make_weight({hs, hs});
        return true;
    }
    if (suffix == "post_attention_layernorm.weight") {
        model->w.mlp_norm_w[layer] = make_weight({hs});
        return true;
    }
    if (suffix == "mlp.gate_proj.weight") {
        model->w.mlp_gate_w[layer] = make_weight({di, hs});
        return true;
    }
    if (suffix == "mlp.up_proj.weight") {
        model->w.mlp_up_w[layer] = make_weight({di, hs});
        return true;
    }
    if (suffix == "mlp.down_proj.weight") {
        model->w.mlp_down_w[layer] = make_weight({hs, di});
        return true;
    }

    return false;
}

int64_t forward_one_token(
    LlaisysQwen2Model *model,
    int64_t token_id,
    size_t pos,
    bool need_next_token,
    int top_k,
    float top_p,
    float temperature) {
    const size_t hs = model->meta.hs;
    const size_t nh = model->meta.nh;
    const size_t nkvh = model->meta.nkvh;
    const size_t dh = model->meta.dh;
    const size_t di = model->meta.di;
    const size_t voc = model->meta.voc;

    tensor_t token_tensor = create_model_tensor(model, {1}, LLAISYS_DTYPE_I64);
    token_tensor->load(&token_id);

    int64_t pos_id = static_cast<int64_t>(pos);
    tensor_t pos_tensor = create_model_tensor(model, {1}, LLAISYS_DTYPE_I64);
    pos_tensor->load(&pos_id);

    // Decoder step input embedding for one token.
    tensor_t hidden = create_model_tensor(model, {1, hs}, model->meta.dtype);
    llaisys::ops::embedding(hidden, token_tensor, model->w.in_embed);

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(dh));
    const size_t nlayer = model->meta.nlayer;
    for (size_t l = 0; l < nlayer; ++l) {
        tensor_t attn_norm = create_model_tensor(model, {1, hs}, model->meta.dtype);
        llaisys::ops::rms_norm(attn_norm, hidden, model->w.attn_norm_w[l], model->meta.epsilon);

        tensor_t q_2d = create_model_tensor(model, {1, nh * dh}, model->meta.dtype);
        tensor_t k_2d = create_model_tensor(model, {1, nkvh * dh}, model->meta.dtype);
        tensor_t v_2d = create_model_tensor(model, {1, nkvh * dh}, model->meta.dtype);
        llaisys::ops::linear(q_2d, attn_norm, model->w.attn_q_w[l], model->w.attn_q_b[l]);
        llaisys::ops::linear(k_2d, attn_norm, model->w.attn_k_w[l], model->w.attn_k_b[l]);
        llaisys::ops::linear(v_2d, attn_norm, model->w.attn_v_w[l], model->w.attn_v_b[l]);

        tensor_t q_3d = q_2d->view({1, nh, dh});
        tensor_t k_3d = k_2d->view({1, nkvh, dh});
        tensor_t v_3d = v_2d->view({1, nkvh, dh});

        tensor_t q_rope = create_model_tensor(model, {1, nh, dh}, model->meta.dtype);
        tensor_t k_rope = create_model_tensor(model, {1, nkvh, dh}, model->meta.dtype);
        llaisys::ops::rope(q_rope, q_3d, pos_tensor, model->meta.theta);
        llaisys::ops::rope(k_rope, k_3d, pos_tensor, model->meta.theta);

        // Append current token's K/V into cache at absolute position `pos`.
        tensor_t k_slot = model->k_cache[l]->slice(0, pos, pos + 1);
        tensor_t v_slot = model->v_cache[l]->slice(0, pos, pos + 1);
        llaisys::ops::rearrange(k_slot, k_rope);
        llaisys::ops::rearrange(v_slot, v_3d);

        // Attention reads all prefix K/V [0, pos] for causal decoding.
        tensor_t k_ctx = model->k_cache[l]->slice(0, 0, pos + 1);
        tensor_t v_ctx = model->v_cache[l]->slice(0, 0, pos + 1);
        tensor_t attn_val = create_model_tensor(model, {1, nh, dh}, model->meta.dtype);
        llaisys::ops::self_attention(attn_val, q_rope, k_ctx, v_ctx, attn_scale);

        tensor_t attn_val_2d = attn_val->view({1, hs});
        tensor_t attn_out = create_model_tensor(model, {1, hs}, model->meta.dtype);
        llaisys::ops::linear(attn_out, attn_val_2d, model->w.attn_o_w[l], nullptr);

        tensor_t residual = create_model_tensor(model, {1, hs}, model->meta.dtype);
        llaisys::ops::add(residual, hidden, attn_out);

        tensor_t mlp_norm = create_model_tensor(model, {1, hs}, model->meta.dtype);
        llaisys::ops::rms_norm(mlp_norm, residual, model->w.mlp_norm_w[l], model->meta.epsilon);

        tensor_t gate = create_model_tensor(model, {1, di}, model->meta.dtype);
        tensor_t up = create_model_tensor(model, {1, di}, model->meta.dtype);
        llaisys::ops::linear(gate, mlp_norm, model->w.mlp_gate_w[l], nullptr);
        llaisys::ops::linear(up, mlp_norm, model->w.mlp_up_w[l], nullptr);

        tensor_t swiglu_out = create_model_tensor(model, {1, di}, model->meta.dtype);
        llaisys::ops::swiglu(swiglu_out, gate, up);

        tensor_t mlp_down = create_model_tensor(model, {1, hs}, model->meta.dtype);
        llaisys::ops::linear(mlp_down, swiglu_out, model->w.mlp_down_w[l], nullptr);

        tensor_t hidden_next = create_model_tensor(model, {1, hs}, model->meta.dtype);
        llaisys::ops::add(hidden_next, residual, mlp_down);
        hidden = std::move(hidden_next);
    }

    tensor_t out_norm = create_model_tensor(model, {1, hs}, model->meta.dtype);
    llaisys::ops::rms_norm(out_norm, hidden, model->w.out_norm_w, model->meta.epsilon);

    tensor_t logits_2d = create_model_tensor(model, {1, voc}, model->meta.dtype);
    llaisys::ops::linear(logits_2d, out_norm, model->w.out_embed, nullptr);

    if (!need_next_token) {
        return -1;
    }

    tensor_t logits_1d = logits_2d->view({voc});
    tensor_t next_idx = create_model_tensor(model, {1}, LLAISYS_DTYPE_I64);

    const bool greedy = (top_k == 1 && top_p >= 1.0f && std::abs(temperature - 1.0f) < 1e-6f);
    if (greedy) {
        tensor_t max_val = create_model_tensor(model, {1}, model->meta.dtype);
        llaisys::ops::argmax(next_idx, max_val, logits_1d);
    } else {
        llaisys::ops::sample(next_idx, logits_1d, temperature, top_k, top_p);
    }
    return reinterpret_cast<const int64_t *>(next_idx->data())[0];
}

int64_t infer_impl(
    LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken,
    int top_k,
    float top_p,
    float temperature) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null.");
    CHECK_ARGUMENT(token_ids != nullptr || ntoken == 0, "Qwen2: token_ids pointer is null.");
    CHECK_ARGUMENT(temperature > 0.0f, "Qwen2: temperature must be > 0.");
    CHECK_ARGUMENT(top_p > 0.0f && top_p <= 1.0f, "Qwen2: top_p must be in (0, 1].");
    ensure_model_device_supported(model);
    check_model_ready(model);

    if (ntoken == 0) {
        return model->meta.end_token;
    }
    CHECK_ARGUMENT(ntoken <= model->meta.maxseq, "Qwen2: ntoken exceeds maxseq.");

    // Reuse KV cache only when the caller keeps a matching prefix.
    bool prefix_match = model->cached_tokens <= ntoken;
    if (prefix_match) {
        for (size_t i = 0; i < model->cached_tokens; ++i) {
            if (model->cached_token_ids[i] != token_ids[i]) {
                prefix_match = false;
                break;
            }
        }
    }
    if (!prefix_match) {
        reset_generation_state(model);
    }

    const bool deterministic = (top_k == 1 && top_p >= 1.0f && std::abs(temperature - 1.0f) < 1e-6f);
    if (deterministic && model->cached_tokens == ntoken && model->cached_next_token_valid && model->cached_next_for_ntoken == ntoken) {
        return model->cached_next_token;
    }

    ensure_kv_capacity(model, ntoken);

    // Process only uncached suffix tokens. Sample only on the final position.
    for (size_t pos = model->cached_tokens; pos < ntoken; ++pos) {
        const bool need_next_token = (pos + 1 == ntoken);
        int64_t next_token = forward_one_token(model, token_ids[pos], pos, need_next_token, top_k, top_p, temperature);
        if (need_next_token) {
            model->cached_next_token = next_token;
            model->cached_next_token_valid = deterministic;
            model->cached_next_for_ntoken = ntoken;
        }

        model->cached_token_ids.push_back(token_ids[pos]);
        model->cached_tokens = pos + 1;
    }

    // If no new prefix token is processed, we still need to produce a next token.
    if (model->cached_tokens == ntoken && (!deterministic || !model->cached_next_token_valid || model->cached_next_for_ntoken != ntoken)) {
        model->cached_next_token = forward_one_token(model, token_ids[ntoken - 1], ntoken - 1, true, top_k, top_p, temperature);
        model->cached_next_token_valid = deterministic;
        model->cached_next_for_ntoken = ntoken;
    }

    CHECK_ARGUMENT(model->cached_next_for_ntoken == ntoken, "Qwen2: failed to produce next token.");
    return model->cached_next_token;
}

} // namespace

__C {
struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    CHECK_ARGUMENT(meta != nullptr, "Qwen2: meta must not be null.");
    CHECK_ARGUMENT(meta->nlayer > 0, "Qwen2: nlayer must be > 0.");
    CHECK_ARGUMENT(meta->hs > 0 && meta->nh > 0 && meta->nkvh > 0, "Qwen2: invalid attention dimensions.");
    CHECK_ARGUMENT(meta->dh > 0 && meta->di > 0 && meta->voc > 0, "Qwen2: invalid model dimensions.");
    CHECK_ARGUMENT(meta->maxseq > 0, "Qwen2: maxseq must be > 0.");
    CHECK_ARGUMENT(meta->hs == meta->nh * meta->dh, "Qwen2: hs must equal nh * dh.");
    CHECK_ARGUMENT(meta->nh % meta->nkvh == 0, "Qwen2: nh must be divisible by nkvh.");
    CHECK_ARGUMENT(is_supported_float_dtype(meta->dtype), "Qwen2: unsupported model dtype.");

    auto *model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device = device;
    model->device_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;

    const size_t nlayer = meta->nlayer;
    model->w.attn_norm_w.resize(nlayer);
    model->w.attn_q_w.resize(nlayer);
    model->w.attn_q_b.resize(nlayer);
    model->w.attn_k_w.resize(nlayer);
    model->w.attn_k_b.resize(nlayer);
    model->w.attn_v_w.resize(nlayer);
    model->w.attn_v_b.resize(nlayer);
    model->w.attn_o_w.resize(nlayer);
    model->w.mlp_norm_w.resize(nlayer);
    model->w.mlp_gate_w.resize(nlayer);
    model->w.mlp_up_w.resize(nlayer);
    model->w.mlp_down_w.resize(nlayer);
    model->k_cache.resize(nlayer);
    model->v_cache.resize(nlayer);

    ensure_model_device_supported(model);
    return model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    (void)model;
    return nullptr;
}

int llaisysQwen2ModelLoadTensor(
    struct LlaisysQwen2Model *model,
    const char *name,
    const void *data,
    size_t *shape,
    size_t ndim,
    llaisysDataType_t dtype) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null.");
    CHECK_ARGUMENT(name != nullptr, "Qwen2: name must not be null.");
    CHECK_ARGUMENT(data != nullptr, "Qwen2: tensor data must not be null.");
    ensure_model_device_supported(model);

    std::vector<size_t> shape_vec = to_shape_vec(shape, ndim);
    return set_named_weight(model, std::string(name), data, shape_vec, dtype) ? 1 : 0;
}

void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null.");
    reset_generation_state(model);
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    return infer_impl(model, token_ids, ntoken, 1, 1.0f, 1.0f);
}

int64_t llaisysQwen2ModelInferEx(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken,
    int top_k,
    float top_p,
    float temperature) {
    return infer_impl(model, token_ids, ntoken, top_k, top_p, temperature);
}
}
