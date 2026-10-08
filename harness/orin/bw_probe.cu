// Orin 实际可持续的显存(统一内存)读带宽探针:用 16 字节向量读一块大缓冲,给 GEMV 类内核的带宽利用率定分母。
// 编译: /usr/local/cuda/bin/nvcc -O3 -arch=sm_87 bw_probe.cu -o bw_probe     运行: ./bw_probe [MiB=2048]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void read_kernel(const uint4 * __restrict__ p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = __ldg(&p[i]);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;  // 防止被优化掉
}

// 模拟 GEMV 的"每线程一次读 4 字节"的窄读(llama.cpp 的 mmvq 内部主要是 4 字节读),看窄读和宽读的差距
__global__ void read_kernel_u32(const unsigned * __restrict__ p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        acc ^= __ldg(&p[i]);
    }
    if (acc == 0x12345678u) out[0] = acc;
}

int main(int argc, char ** argv) {
    size_t mib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2048;
    size_t bytes = mib << 20;
    void * buf = nullptr;
    unsigned * out = nullptr;
    if (cudaMalloc(&buf, bytes) != cudaSuccess || cudaMalloc(&out, 16) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
    cudaMemset(buf, 1, bytes);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int mode = 0; mode < 2; mode++) {
        for (int blocks : {64, 128, 256, 512, 1024, 2048}) {
            for (int threads : {128, 256}) {
                float best = 1e30f;
                for (int it = 0; it < 12; it++) {
                    cudaEventRecord(a);
                    if (mode == 0) read_kernel<<<blocks, threads>>>((const uint4 *) buf, bytes / 16, out);
                    else           read_kernel_u32<<<blocks, threads>>>((const unsigned *) buf, bytes / 4, out);
                    cudaEventRecord(b);
                    cudaEventSynchronize(b);
                    float ms;
                    cudaEventElapsedTime(&ms, a, b);
                    if (it >= 2 && ms < best) best = ms;
                }
                printf("%s blocks=%4d threads=%3d : %7.2f ms  %7.1f GB/s\n", mode == 0 ? "16B读" : " 4B读", blocks, threads, best, bytes / 1e9 / (best / 1e3));
            }
        }
    }
    return 0;
}
