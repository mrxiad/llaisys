#include "../runtime_api.hpp"

#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

namespace runtime_api {
namespace {

void check_cuda(cudaError_t code, const char *api_name) {
    if (code == cudaSuccess) {
        return;
    }
    std::cerr << "[ERROR] CUDA runtime call failed: " << api_name
              << " error=" << cudaGetErrorString(code) << EXCEPTION_LOCATION_MSG << std::endl;
    throw std::runtime_error("CUDA runtime call failed");
}

cudaMemcpyKind to_cuda_memcpy_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        CHECK_ARGUMENT(false, "Unsupported memcpy kind.");
        return cudaMemcpyDefault;
    }
}

} // namespace

int getDeviceCount() {
    int ndev = 0;
    cudaError_t code = cudaGetDeviceCount(&ndev);
    if (code == cudaSuccess) {
        return ndev;
    }
    if (code == cudaErrorNoDevice || code == cudaErrorInsufficientDriver) {
        (void)cudaGetLastError(); // clear sticky error state for later runtime calls
        return 0;
    }
    check_cuda(code, "cudaGetDeviceCount");
    return 0;
}

void setDevice(int device_id) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
}

void deviceSynchronize() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    check_cuda(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
}
void streamSynchronize(llaisysStream_t stream) {
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    check_cuda(cudaFree(ptr), "cudaFree");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    check_cuda(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    check_cuda(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    check_cuda(cudaMemcpy(dst, src, size, to_cuda_memcpy_kind(kind)), "cudaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    check_cuda(
        cudaMemcpyAsync(
            dst,
            src,
            size,
            to_cuda_memcpy_kind(kind),
            reinterpret_cast<cudaStream_t>(stream)),
        "cudaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
