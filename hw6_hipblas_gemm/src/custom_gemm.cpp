// 手写 SGEMM，行主序，C = A * B。
//
//   1) 每个 block 处理 BM x BN 的 C tile。
//   2) 每个 thread 计算 TM x TN 个 C 元素（寄存器分块，提高算术强度）。
//   3) 循环 K 方向：把 A 的 BM x BK 和 B 的 BK x BN 分别搬进 shared memory，
//      然后在 shared memory 里做 outer product 累加到寄存器。
//
// 参数：BM=128, BN=128, BK=16, TM=8, TN=8 → 每个 block 有 (BM/TM) * (BN/TN)
// = 16 * 16 = 256 个线程，每个线程算 64 个 C 元素。
//
// 这份实现故意写得可读，不做 double buffering / async copy，让重点停留在
// tiling 思想上。要提性能可以进一步加：async prefetch、bank conflict 消除、
// vectorized load (float4)、更大的 TM/TN + 更宽的 K 展开。

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#define HIP_CHECK(x)                                                          \
  do {                                                                        \
    hipError_t e = (x);                                                       \
    if (e != hipSuccess) {                                                    \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                         \
                   hipGetErrorString(e), __FILE__, __LINE__);                 \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS_PER_BLOCK = (BM / TM) * (BN / TN);  // 256

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void sgemm_tiled(int M, int N, int K,
                 const float * __restrict__ A,
                 const float * __restrict__ B,
                 float       * __restrict__ C) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  const int tid = threadIdx.y * (BN / TN) + threadIdx.x;
  // 每个线程负责的 C 子块起点
  const int thread_row = threadIdx.y * TM;
  const int thread_col = threadIdx.x * TN;

  // 累加寄存器
  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  // 协作搬运：256 线程一次要搬 BM*BK=2048 个 A 元素、BK*BN=2048 个 B 元素
  // 每线程搬 8 个（这里为了简单每线程 1 次搬 1 元素 * 8 循环，方便读）
  constexpr int A_TILE_ELEMS = BM * BK;  // 2048
  constexpr int B_TILE_ELEMS = BK * BN;  // 2048
  constexpr int PER_THREAD_A = A_TILE_ELEMS / THREADS_PER_BLOCK;  // 8
  constexpr int PER_THREAD_B = B_TILE_ELEMS / THREADS_PER_BLOCK;  // 8

  for (int k0 = 0; k0 < K; k0 += BK) {
    // 加载 A tile: A[block_row : block_row+BM, k0 : k0+BK]
    #pragma unroll
    for (int i = 0; i < PER_THREAD_A; ++i) {
      int idx = tid + i * THREADS_PER_BLOCK;
      int r = idx / BK;
      int c = idx % BK;
      int gr = block_row + r;
      int gc = k0 + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
    }
    // 加载 B tile: B[k0 : k0+BK, block_col : block_col+BN]
    #pragma unroll
    for (int i = 0; i < PER_THREAD_B; ++i) {
      int idx = tid + i * THREADS_PER_BLOCK;
      int r = idx / BN;
      int c = idx % BN;
      int gr = k0 + r;
      int gc = block_col + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
    }
    __syncthreads();

    // 在 shared 上做 outer product 累加到寄存器
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_reg[TM], b_reg[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) a_reg[i] = As[thread_row + i][kk];
      #pragma unroll
      for (int j = 0; j < TN; ++j) b_reg[j] = Bs[kk][thread_col + j];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += a_reg[i] * b_reg[j];
    }
    __syncthreads();
  }

  // 写回 C
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    int gr = block_row + thread_row + i;
    if (gr >= M) break;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      int gc = block_col + thread_col + j;
      if (gc < N) C[gr * N + gc] = acc[i][j];
    }
  }
}

static void cpu_gemm(int M, int N, int K,
                     const float *A, const float *B, float *C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

int main(int argc, char **argv) {
  int M = 1024, N = 1024, K = 1024, iters = 20;
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) iters = std::atoi(argv[4]);
  std::printf("custom SGEMM: M=%d N=%d K=%d iters=%d  BM=%d BN=%d BK=%d TM=%d TN=%d\n",
              M, N, K, iters, BM, BN, BK, TM, TN);

  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    std::fprintf(stderr, "为简化，要求 M%%BM==0, N%%BN==0, K%%BK==0\n");
    return 1;
  }

  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto &v : hA) v = dist(rng);
  for (auto &v : hB) v = dist(rng);

  float *dA, *dB, *dC;
  HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dC, hC.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice));

  dim3 block(BN / TN, BM / TM);   // 16 x 16 = 256
  dim3 grid(N / BN, M / BM);

  // warmup
  for (int i = 0; i < 3; ++i)
    hipLaunchKernelGGL(sgemm_tiled, grid, block, 0, 0, M, N, K, dA, dB, dC);
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t s, e;
  hipEventCreate(&s); hipEventCreate(&e);
  hipEventRecord(s);
  for (int i = 0; i < iters; ++i)
    hipLaunchKernelGGL(sgemm_tiled, grid, block, 0, 0, M, N, K, dA, dB, dC);
  hipEventRecord(e);
  hipEventSynchronize(e);
  float ms = 0.f;
  hipEventElapsedTime(&ms, s, e);
  double per_ms = ms / iters;
  double gflops = 2.0 * M * N * K / (per_ms * 1e6);
  std::printf("custom   avg %.3f ms  %.1f GFLOPS\n", per_ms, gflops);

  if ((size_t)M * N * K <= 512ull * 512 * 512) {
    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> ref((size_t)M * N, 0.f);
    cpu_gemm(M, N, K, hA.data(), hB.data(), ref.data());
    double max_err = 0.0, ref_max = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      double d = std::abs((double)ref[i] - (double)hC[i]);
      if (d > max_err) max_err = d;
      if (std::abs((double)ref[i]) > ref_max) ref_max = std::abs((double)ref[i]);
    }
    std::printf("custom   max abs err = %g (ref max = %g, rel = %g)\n",
                max_err, ref_max, max_err / ref_max);
  }

  hipFree(dA); hipFree(dB); hipFree(dC);
  hipEventDestroy(s); hipEventDestroy(e);
  return 0;
}
