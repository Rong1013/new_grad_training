
## 目录内容

| 文件 | 说明 |
|---|---|
| `reduce.h` | 对外接口声明，`reduce_sum(d_in, d_out, n)` |
| `reduce.cpp` | HIP kernel 实现（grid-stride loop + warp shuffle + shared memory + atomicAdd） |
| `test_reduce.cpp` | 单元测试，对比 GPU FP32 结果与 CPU FP64 参考值 |
| `CMakeLists.txt` | 使用 hipcc 编译 `.cpp` 文件的构建脚本 |

## 编译与运行

```bash
cd /workspace/new_grad_training/hw1_cuda_reduce
cmake -B build && cmake --build build
./build/test_reduce
```

`CMakeLists.txt` 中已经通过 `set(CMAKE_CXX_COMPILER hipcc CACHE STRING ...)` 指定 hipcc 作为编译器，无需在命令行再传 `-DCMAKE_CXX_COMPILER=hipcc`。

预期输出：

```
GPU=xxxxx.xx  CPU=xxxxx.xx  rel_err=x.xxe-06  PASS
```

## 算法设计概览

Reduce 的本质：把 O(N) 的串行加法转成 O(log N) 的层级并行归约。沿 GPU 的执行层级 **grid → block → warp → thread** 自底向上逐层归并：

```
全局数据 (N elements)
    ↓ grid-stride loop：每个 thread 累加多个元素
Thread 局部和
    ↓ warp shuffle：warp 内 64 线程规约
Warp 部分和 (BLOCK_SIZE/64 个 / block)
    ↓ shared memory + 第二轮 warp shuffle
Block 部分和 (blocks 个)
    ↓ atomicAdd：跨 block 汇总到全局输出
最终标量 (1 个)
```

## 关键优化点

1. **Grid-stride loop**：固定启动 ≤1024 个 block，用跨步循环吃下任意大小 N，兼顾合并访存与低 launch 开销。
2. **Warp shuffle**：`__shfl_down` 在寄存器层面完成 warp 内规约，省去 SMEM 存取、bank conflict 和 `__syncthreads`。
3. **两级层次结构**：warp 内 shuffle → SMEM 存部分和 → 首个 warp 再做一次 shuffle，仅需一次 `__syncthreads()`。
4. **单次 atomicAdd**：每个 block 只做一次原子加，避免二阶段 kernel 的中间 buffer 与额外 launch 开销。

## 精度分析

- FP32 加法相对误差 ε ≈ 6e-8
- 朴素串行累加最坏误差 O(N·ε)，N=4M 时可达 2.5e-1，不合格
- 树形规约误差 O(log₂ N · ε)，N=4M 时约 1.3e-6
- 远优于双万分之一（2e-4）的验收标准

## 平台差异说明

- AMD GPU 的 warp size = 64（NVIDIA = 32），本实现按 64 编写：`warp_reduce` 的 offset 从 32 开始、SMEM 大小为 `BLOCK_SIZE / 64`。
- 如果要在 NVIDIA CUDA 平台运行，需要把 warp size 相关常量改回 32，并把 `__shfl_down` 换为 `__shfl_down_sync(0xffffffff, ...)`。

## 可扩展方向

1. 向量化访存（`float4`）减少 load 指令数
2. Kahan / pairwise 求和进一步降低数值误差
3. 模板化支持 `sum / max / min / prod` 等其他二元运算
4. 大规模场景下改为两阶段 kernel，规避 atomicAdd 竞争
