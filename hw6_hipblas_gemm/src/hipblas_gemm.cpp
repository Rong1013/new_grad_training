// hipBLAS SGEMM demo:
//   C = alpha * A * B + beta * C
// A: MxK, B: KxN, C: MxN，全部行主序（我们通过参数把 hipBLAS 的列主序适配过来）
//
// 关键点：
//   1) 参考实现在 CPU 上算一遍，用来对答案。
//   2) 用 hipEvent_t 做 GPU 侧计时，报告 GFLOPs。
//   3) hipBLAS 默认是列主序 (Fortran)。我们的输入是行主序，用一个常见技巧：
//        C_row = A_row * B_row
//      等价于
//        C_col^T = B_col * A_col   （因为 X_row = X_col^T）
//      所以调用 hipblasSgemm(op_N, op_N, N, M, K, alpha, B, N, A, K, beta, C, N)
//      即可，不用真的做 transpose。

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>

#define HIP_CHECK(x)                                                          \
  do {                                                                        \
    hipError_t e = (x);                                                       \
    if (e != hipSuccess) {                                                    \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                         \
                   hipGetErrorString(e), __FILE__, __LINE__);                 \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define HIPBLAS_CHECK(x)                                                      \
  do {                                                                        \
    hipblasStatus_t s = (x);                                                  \
    if (s != HIPBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "hipBLAS error %d at %s:%d\n",                     \
                   (int)s, __FILE__, __LINE__);                               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static void cpu_gemm(int M, int N, int K, float alpha,
                     const float *A, const float *B, float beta, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = alpha * s + beta * C[i * N + j];
    }
  }
}

int main(int argc, char **argv) {
  int M = 1024, N = 1024, K = 1024;
  int iters = 20;
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) iters = std::atoi(argv[4]);
  std::printf("hipBLAS SGEMM: M=%d N=%d K=%d iters=%d\n", M, N, K, iters);

  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N, 0.f);
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
  HIP_CHECK(hipMemset(dC, 0, hC.size() * sizeof(float)));

  hipblasHandle_t handle;
  HIPBLAS_CHECK(hipblasCreate(&handle));

  const float alpha = 1.f, beta = 0.f;

  // warmup
  for (int i = 0; i < 3; ++i) {
    HIPBLAS_CHECK(hipblasSgemm(handle,
                               HIPBLAS_OP_N, HIPBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               dB, N,
                               dA, K,
                               &beta,
                               dC, N));
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t s, e;
  hipEventCreate(&s);
  hipEventCreate(&e);
  hipEventRecord(s, 0);
  for (int i = 0; i < iters; ++i) {
    HIPBLAS_CHECK(hipblasSgemm(handle,
                               HIPBLAS_OP_N, HIPBLAS_OP_N,
                               N, M, K,
                               &alpha, dB, N,
                                       dA, K,
                               &beta,  dC, N));
  }
  hipEventRecord(e, 0);
  hipEventSynchronize(e);
  float ms = 0.f;
  hipEventElapsedTime(&ms, s, e);
  double per_ms = ms / iters;
  double gflops = 2.0 * M * N * K / (per_ms * 1e6);
  std::printf("hipBLAS  avg %.3f ms  %.1f GFLOPS\n", per_ms, gflops);

  // 数据规模不大时做正确性校验
  if ((size_t)M * N * K <= 512ull * 512 * 512) {
    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> ref((size_t)M * N, 0.f);
    cpu_gemm(M, N, K, alpha, hA.data(), hB.data(), beta, ref.data());
    double max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      double d = std::abs((double)ref[i] - (double)hC[i]);
      if (d > max_err) max_err = d;
    }
    std::printf("hipBLAS  max abs err vs cpu = %g\n", max_err);
  }

  HIPBLAS_CHECK(hipblasDestroy(handle));
  HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
  hipEventDestroy(s); hipEventDestroy(e);
  return 0;
}
