#include "reduce.h"
#include <hip/hip_runtime.h>

// 每个 block 的线程数，必须是 warp size(64) 的整数倍
#define BLOCK_SIZE 256

// warp 内规约：利用 __shfl_down 在寄存器层面完成 warp 内求和，无需 shared memory
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);
    return val;
}

// 主规约 kernel
// 每个 block 先用 grid-stride loop 将全局数据累加到线程局部变量，
// 再经 warp 规约 -> shared memory -> 第二轮 warp 规约，最后 atomicAdd 到全局输出
__global__ void reduce_kernel(const float* __restrict__ in, float* out, int n) {
    // 每个 warp 的规约结果暂存在 shared memory
    // BLOCK_SIZE / 64 = warp 数量（AMD warp size = 64）
    __shared__ float smem[BLOCK_SIZE / 64];

    int tid = threadIdx.x;
    float sum = 0.f;

    // grid-stride loop：每个线程跨步累加，支持 n 远大于 gridDim.x * blockDim.x 的情况
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        sum += in[i];

    // 第一轮：warp 内规约，每个 warp 得到一个部分和
    sum = warp_reduce(sum);

    // 每个 warp 的 lane 0 将结果写入 shared memory
    if (tid % 64 == 0)
        smem[tid / 64] = sum;
    __syncthreads();

    // 第二轮：用第一个 warp 对 shared memory 中的部分和再做一次 warp 规约
    if (tid < BLOCK_SIZE / 64)
        sum = warp_reduce(smem[tid]);

    // block 内 thread 0 将本 block 的最终结果原子加到全局输出
    if (tid == 0)
        atomicAdd(out, sum);
}

void reduce_sum(const float* d_in, float* d_out, int n) {
    // block 数量上限 1024，避免启动过多 block 导致调度开销
    int blocks = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);
    // 清零输出，保证 atomicAdd 从 0 开始累加
    hipMemset(d_out, 0, sizeof(float));
    hipLaunchKernelGGL(reduce_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, 0, d_in, d_out, n);
}
