#include "op_nvidia.hpp"

#include "../../utils.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace llaisys::ops::nvidia {
namespace {

void check_cuda(cudaError_t code, const char *api_name) {
    if (code == cudaSuccess) {
        return;
    }
    std::cerr << "[ERROR] CUDA call failed: " << api_name
              << " error=" << cudaGetErrorString(code) << EXCEPTION_LOCATION_MSG << std::endl;
    throw std::runtime_error("CUDA call failed");
}

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float to_float<fp16_t>(fp16_t x) {
    return __half2float(__ushort_as_half(x._v));
}

template <>
__device__ __forceinline__ float to_float<bf16_t>(bf16_t x) {
    uint32_t bits = static_cast<uint32_t>(x._v) << 16;
    return __uint_as_float(bits);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ fp16_t from_float<fp16_t>(float x) {
    fp16_t out{};
    out._v = __half_as_ushort(__float2half_rn(x));
    return out;
}

template <>
__device__ __forceinline__ bf16_t from_float<bf16_t>(float x) {
    // round-to-nearest-even on float->bf16 conversion
    uint32_t bits = __float_as_uint(x);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    bf16_t out{};
    out._v = static_cast<uint16_t>(bits >> 16);
    return out;
}

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    c[idx] = from_float<T>(to_float<T>(a[idx]) + to_float<T>(b[idx]));
}

template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    __shared__ float best_val[256];
    __shared__ int64_t best_idx[256];

    const int tid = threadIdx.x;
    float local_best_val = -INFINITY;
    int64_t local_best_idx = 0;

    for (size_t i = static_cast<size_t>(tid); i < n; i += blockDim.x) {
        const float v = to_float<T>(vals[i]);
        if (v > local_best_val) {
            local_best_val = v;
            local_best_idx = static_cast<int64_t>(i);
        }
    }

    best_val[tid] = local_best_val;
    best_idx[tid] = local_best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (best_val[tid + stride] > best_val[tid]) {
                best_val[tid] = best_val[tid + stride];
                best_idx[tid] = best_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = best_idx[0];
        max_val[0] = from_float<T>(best_val[0]);
    }
}

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, size_t nindex, size_t nembed) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = nindex * nembed;
    if (idx >= total) {
        return;
    }
    const size_t row = idx / nembed;
    const size_t col = idx % nembed;
    const size_t src_row = static_cast<size_t>(index[row]);
    out[idx] = weight[src_row * nembed + col];
}

template <typename T>
__global__ void linear_kernel(T *out,
                              const T *in,
                              const T *weight,
                              const T *bias,
                              size_t m,
                              size_t n,
                              size_t k,
                              int has_bias) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = m * n;
    if (idx >= total) {
        return;
    }
    const size_t row = idx / n;
    const size_t col = idx % n;

    float acc = has_bias ? to_float<T>(bias[col]) : 0.0f;
    const size_t in_off = row * k;
    const size_t w_off = col * k;
    for (size_t j = 0; j < k; ++j) {
        acc += to_float<T>(in[in_off + j]) * to_float<T>(weight[w_off + j]);
    }
    out[idx] = from_float<T>(acc);
}

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t nrow, size_t ncol, float eps) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= nrow) {
        return;
    }

    __shared__ float sum_shared[256];
    const int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (size_t j = static_cast<size_t>(tid); j < ncol; j += blockDim.x) {
        const float x = to_float<T>(in[row * ncol + j]);
        local_sum += x * x;
    }
    sum_shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(sum_shared[0] / static_cast<float>(ncol) + eps);
    for (size_t j = static_cast<size_t>(tid); j < ncol; j += blockDim.x) {
        const float y = to_float<T>(in[row * ncol + j]) * inv_rms * to_float<T>(weight[j]);
        out[row * ncol + j] = from_float<T>(y);
    }
}

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t d, float theta) {
    const size_t half = d / 2;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = seqlen * nhead * half;
    if (idx >= total) {
        return;
    }

    const size_t j = idx % half;
    const size_t tmp = idx / half;
    const size_t h = tmp % nhead;
    const size_t i = tmp / nhead;

    const float pos = static_cast<float>(pos_ids[i]);
    const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(d);
    const double freq = static_cast<double>(pos) / pow(static_cast<double>(theta), static_cast<double>(exponent));
    const double c = cos(freq);
    const double s = sin(freq);

    const size_t base = (i * nhead + h) * d;
    const float a = to_float<T>(in[base + j]);
    const float b = to_float<T>(in[base + half + j]);
    const double out_a = fma(-static_cast<double>(b), s, static_cast<double>(a) * c);
    const double out_b = fma(static_cast<double>(a), s, static_cast<double>(b) * c);
    out[base + j] = from_float<T>(static_cast<float>(out_a));
    out[base + half + j] = from_float<T>(static_cast<float>(out_b));
}

template <typename T>
__global__ void self_attention_kernel(T *attn_val,
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
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = qlen * nh * dv;
    if (idx >= total) {
        return;
    }

    const size_t j = idx % dv;
    const size_t tmp = idx / dv;
    const size_t h = tmp % nh;
    const size_t t = tmp / nh;

    const size_t group = nh / nkvh;
    const size_t kvh = h / group;
    const size_t valid_kv = kvlen - qlen + t + 1;

    float max_score = -INFINITY;
    for (size_t s = 0; s < valid_kv; ++s) {
        const size_t q_base = (t * nh + h) * d;
        const size_t k_base = (s * nkvh + kvh) * d;
        float dot = 0.0f;
        for (size_t x = 0; x < d; ++x) {
            dot += to_float<T>(q[q_base + x]) * to_float<T>(k[k_base + x]);
        }
        const float score = dot * scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    float sum_exp = 0.0f;
    float out_val = 0.0f;
    for (size_t s = 0; s < valid_kv; ++s) {
        const size_t q_base = (t * nh + h) * d;
        const size_t k_base = (s * nkvh + kvh) * d;
        float dot = 0.0f;
        for (size_t x = 0; x < d; ++x) {
            dot += to_float<T>(q[q_base + x]) * to_float<T>(k[k_base + x]);
        }
        const float p = expf(dot * scale - max_score);
        sum_exp += p;
        const size_t v_base = (s * nkvh + kvh) * dv;
        out_val += p * to_float<T>(v[v_base + j]);
    }

    attn_val[idx] = from_float<T>(out_val / sum_exp);
}

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const float g = to_float<T>(gate[idx]);
    const float val = to_float<T>(up[idx]) * (g / (1.0f + expf(-g)));
    out[idx] = from_float<T>(val);
}

template <typename T>
void launch_add(tensor_t c, tensor_t a, tensor_t b) {
    const int threads = 256;
    const int blocks = static_cast<int>((c->numel() + threads - 1) / threads);
    add_kernel<T><<<blocks, threads>>>(
        reinterpret_cast<T *>(c->data()),
        reinterpret_cast<const T *>(a->data()),
        reinterpret_cast<const T *>(b->data()),
        c->numel());
}

template <typename T>
void launch_argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    argmax_kernel<T><<<1, 256>>>(
        reinterpret_cast<int64_t *>(max_idx->data()),
        reinterpret_cast<T *>(max_val->data()),
        reinterpret_cast<const T *>(vals->data()),
        vals->numel());
}

template <typename T>
void launch_embedding(tensor_t out, tensor_t index, tensor_t weight) {
    const size_t total = out->numel();
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    embedding_kernel<T><<<blocks, threads>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const int64_t *>(index->data()),
        reinterpret_cast<const T *>(weight->data()),
        index->shape()[0],
        weight->shape()[1]);
}

template <typename T>
void launch_linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const bool has_bias = bias != nullptr;
    const size_t m = in->shape()[0];
    const size_t k = in->shape()[1];
    const size_t n = weight->shape()[0];

    const size_t total = out->numel();
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    linear_kernel<T><<<blocks, threads>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(in->data()),
        reinterpret_cast<const T *>(weight->data()),
        has_bias ? reinterpret_cast<const T *>(bias->data()) : nullptr,
        m,
        n,
        k,
        has_bias ? 1 : 0);
}

template <typename T>
void launch_rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    const int threads = 256;
    const int blocks = static_cast<int>(in->shape()[0]);
    rms_norm_kernel<T><<<blocks, threads>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(in->data()),
        reinterpret_cast<const T *>(weight->data()),
        in->shape()[0],
        in->shape()[1],
        eps);
}

template <typename T>
void launch_rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    const size_t total = in->shape()[0] * in->shape()[1] * (in->shape()[2] / 2);
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    rope_kernel<T><<<blocks, threads>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(in->data()),
        reinterpret_cast<const int64_t *>(pos_ids->data()),
        in->shape()[0],
        in->shape()[1],
        in->shape()[2],
        theta);
}

template <typename T>
void launch_self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    const size_t total = attn_val->numel();
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    self_attention_kernel<T><<<blocks, threads>>>(
        reinterpret_cast<T *>(attn_val->data()),
        reinterpret_cast<const T *>(q->data()),
        reinterpret_cast<const T *>(k->data()),
        reinterpret_cast<const T *>(v->data()),
        q->shape()[0],
        k->shape()[0],
        q->shape()[1],
        k->shape()[1],
        q->shape()[2],
        v->shape()[2],
        scale);
}

template <typename T>
void launch_swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    const int threads = 256;
    const int blocks = static_cast<int>((out->numel() + threads - 1) / threads);
    swiglu_kernel<T><<<blocks, threads>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(gate->data()),
        reinterpret_cast<const T *>(up->data()),
        out->numel());
}

} // namespace

void add(tensor_t c, tensor_t a, tensor_t b) {
    switch (c->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_add<float>(c, a, b);
        break;
    case LLAISYS_DTYPE_F16:
        launch_add<fp16_t>(c, a, b);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_add<bf16_t>(c, a, b);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(c->dtype());
    }
    check_cuda(cudaGetLastError(), "add_kernel");
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_argmax<float>(max_idx, max_val, vals);
        break;
    case LLAISYS_DTYPE_F16:
        launch_argmax<fp16_t>(max_idx, max_val, vals);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_argmax<bf16_t>(max_idx, max_val, vals);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
    check_cuda(cudaGetLastError(), "argmax_kernel");
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_embedding<float>(out, index, weight);
        break;
    case LLAISYS_DTYPE_F16:
        launch_embedding<fp16_t>(out, index, weight);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_embedding<bf16_t>(out, index, weight);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
    check_cuda(cudaGetLastError(), "embedding_kernel");
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_linear<float>(out, in, weight, bias);
        break;
    case LLAISYS_DTYPE_F16:
        launch_linear<fp16_t>(out, in, weight, bias);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_linear<bf16_t>(out, in, weight, bias);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
    check_cuda(cudaGetLastError(), "linear_kernel");
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_rms_norm<float>(out, in, weight, eps);
        break;
    case LLAISYS_DTYPE_F16:
        launch_rms_norm<fp16_t>(out, in, weight, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_rms_norm<bf16_t>(out, in, weight, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
    check_cuda(cudaGetLastError(), "rms_norm_kernel");
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_rope<float>(out, in, pos_ids, theta);
        break;
    case LLAISYS_DTYPE_F16:
        launch_rope<fp16_t>(out, in, pos_ids, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_rope<bf16_t>(out, in, pos_ids, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
    check_cuda(cudaGetLastError(), "rope_kernel");
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_self_attention<float>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_F16:
        launch_self_attention<fp16_t>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_self_attention<bf16_t>(attn_val, q, k, v, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
    check_cuda(cudaGetLastError(), "self_attention_kernel");
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        launch_swiglu<float>(out, gate, up);
        break;
    case LLAISYS_DTYPE_F16:
        launch_swiglu<fp16_t>(out, gate, up);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_swiglu<bf16_t>(out, gate, up);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
    check_cuda(cudaGetLastError(), "swiglu_kernel");
}

} // namespace llaisys::ops::nvidia
