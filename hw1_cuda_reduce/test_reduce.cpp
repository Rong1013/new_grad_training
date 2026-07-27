#include "reduce.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    // 测试规模：4M 个元素，足以覆盖多 block 路径
    const int N = 1 << 22;

    // 在主机端生成随机输入，同时用 double 累加得到高精度参考值
    float* h_in = new float[N];
    double ref = 0.0;
    srand(42);
    for (int i = 0; i < N; i++) {
        h_in[i] = (rand() % 200 - 100) / 100.f;
        ref += h_in[i];  // double 累加避免参考值本身精度损失
    }

    // 分配设备内存
    float *d_in, *d_out, h_out;
    hipMalloc(&d_in, N * sizeof(float));
    hipMalloc(&d_out, sizeof(float));

    // 将输入数据拷贝到设备
    hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice);

    // 调用 HIP reduce 算子
    reduce_sum(d_in, d_out, N);
    hipDeviceSynchronize();

    // 将结果拷回主机
    hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost);

    // 计算相对误差
    double rel_err = fabs((double)h_out - ref) / (fabs(ref) + 1e-9);
    printf("GPU=%.6f  CPU=%.6f  rel_err=%.2e  %s\n",
           h_out, (float)ref, rel_err,
           rel_err < 2e-4 ? "PASS" : "FAIL");

    delete[] h_in;
    hipFree(d_in);
    hipFree(d_out);
    return rel_err < 2e-4 ? 0 : 1;
}
