# hipBLAS GEMM demo + 手写 GEMM kernel + CMake + hipprof 性能分析

## 一、目录结构

```
hw6_hipblas_gemm/
├── CMakeLists.txt          # CMake 配置，同时编两个 exe
├── src/
│   ├── hipblas_gemm.cpp    # 用 hipBLAS 库的 SGEMM demo，做对答案
│   └── custom_gemm.cpp     # 手写 tiled SGEMM kernel（BM128 BN128 BK16 TM8 TN8）
└── build/                  # cmake 输出目录（含 hipprof trace）
```

## 二、构建

```bash
cd /workspace/new_grad_training/hw6_hipblas_gemm
mkdir -p build && cd build
cmake ..                        # 配置阶段，读 CMakeLists.txt 生成 Makefile
make -j                         # 构建阶段
make run                        # 跑两个 exe 对比性能
make profile                    # 跑 hipprof 收集 trace，落到 build/*.json
```

CMake 基本套路可以就此记住这几步：

- **命令 = 配置 (`cmake ..`) + 构建 (`make -j`)**。配置阶段读 CMakeLists，生成 Makefile / ninja.build；构建阶段真正编。
- **变量**：`set(FOO xxx)`、`option(BAR ...)`、`-DKEY=VAL` 命令行覆盖。这里我们用 `-DGPU_ARCHS=gfx936;gfx928` 就能一次编多架构。
- **依赖表达**：`add_executable`/`add_library`、`target_link_libraries`、`target_include_directories`。target-based 写法（带 PRIVATE/PUBLIC/INTERFACE）比早年的全局 `include_directories` 干净。
- **找第三方库**：`find_package(hip)` / `find_package(hipblas)` 是官方推荐做法，靠 `CMAKE_PREFIX_PATH` 指到 `/opt/dtk` 就能找到。为了减少版本差异，本 demo 直接用 `target_include_directories(/opt/dtk/include)` + `target_link_libraries(hipblas amdhip64)`，够用。
- **切编译器**：`-DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc`，或者像这里在 CMakeLists 里 default 一个。hipcc 本质是 clang 前端，加了 offload flag。
- **自定义 target**：`add_custom_target(run ...)`、`add_custom_target(profile ...)`，直接把常用运行/分析命令挂到 `make run`、`make profile`，避免每次手敲。
- **build type**：`-DCMAKE_BUILD_TYPE=Release` / `Debug` / `RelWithDebInfo`，控制 `-O3 -DNDEBUG` vs `-O0 -g`。

CMakeLists.txt 关键片段：

```cmake
set(CMAKE_CXX_COMPILER "/opt/dtk/bin/hipcc" CACHE FILEPATH "" FORCE)
project(hw6_gemm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_PREFIX_PATH "/opt/dtk" "/opt/dtk/hip" "/opt/dtk/hipblas")
foreach(arch ${GPU_ARCHS})
  add_compile_options(--offload-arch=${arch})
  add_link_options(--offload-arch=${arch})
endforeach()
add_executable(hipblas_gemm src/hipblas_gemm.cpp)
target_link_libraries(hipblas_gemm PRIVATE hipblas amdhip64)
add_executable(custom_gemm src/custom_gemm.cpp)
target_link_libraries(custom_gemm PRIVATE amdhip64)
```

## 三、hipBLAS SGEMM demo

代码：[src/hipblas_gemm.cpp](src/hipblas_gemm.cpp)。要点：

- `C = alpha*A*B + beta*C`，输入按行主序生成，然后调 `hipblasSgemm` 时用 **交换 A/B、交换 M/N** 的经典技巧适配 hipBLAS 的列主序，不用真的做 transpose。
- 用 `hipEvent_t` 计时（HIP 事件比 `chrono` 更贴近 GPU 真实时长）。
- 小规模跑 CPU 参考实现校验 `max abs err`。

实测（M=N=K=1024, iters=30, gfx936）：

```
hipBLAS  avg 0.058 ms  37172.6 GFLOPS
hipBLAS  max abs err vs cpu = 4.19e-05
```

## 四、手写 tiled SGEMM kernel（三段式）

代码：[src/custom_gemm.cpp](src/custom_gemm.cpp)。参考"GPU 编程最佳实践"里的经典分块：

- **Block tile**：每个 block 处理 C 的 `BM × BN = 128 × 128` 子块。
- **Thread tile**：每个 thread 计算 `TM × TN = 8 × 8` 个 C 元素，累加寄存器 `acc[TM][TN]`。这样一个 block 有 `(BM/TM) * (BN/TN) = 256` 线程。
- **Shared memory 分块 + 流式 K**：循环 `k0 += BK`，把 `A[BM,BK]` 和 `B[BK,BN]` 协作搬进 `__shared__`，再在 shared 上做 outer product，累加到寄存器。

为什么快 —— 三层内存的算术强度阶梯：

- **不分块**：每个 C 元素要读 2K 个全局内存，算 2K 次浮点，AI ≈ 1 FLOP/word，明显是 memory-bound。
- **只共享内存分块**：每次 tile 从 global 读 `2*BM*BK` 个 word，产出 `BM*BN*BK` 个 FMA，AI 提高到 `BM*BN / (BM+BN)`。
- **再加寄存器分块**：outer product 让每个 shared load 复用 TM/TN 倍，AI ≈ `TM*TN/(TM+TN)` 倍进一步放大。这才是 GPU 上手写 GEMM 能上多个 TFLOPs 的关键。

实测：

```
custom   avg 0.697 ms  3081.2 GFLOPS
custom   max abs err = 0.00025 (ref max ≈ 27，相对 ~1e-5)
```

大约是 hipBLAS 的 8%。这个 baseline 还没做：double buffer、`float4` 向量化 load、K 方向 unroll 2x、shared memory bank conflict 消除、`__builtin_amdgcn_s_barrier` 精细控制。加上这些通常能到 hipBLAS 的 60-80%。想再进一步就得写 MFMA/WMMA 汇编。

## 五、用 hipprof 做性能分析

hipprof 是 DTK 自带的 profiler，把 `hipprof` 前缀加在 exe 前面就能采集。常用开关：

- `--hip-trace`：HIP API + kernel 时间线。
- `--stats`：只打印汇总统计（省磁盘）。
- `--output-type 0|1|2`：0=json、1=html、2=perfetto（默认）。
- `-o <前缀>`：输出文件前缀。

我们做的两次采集（`make profile` 已经封好了）：

```bash
/opt/dtk/bin/hipprof --hip-trace --stats --output-type 0 \
  -o build/hipblas_prof ./build/hipblas_gemm 1024 1024 1024 30
/opt/dtk/bin/hipprof --hip-trace --stats --output-type 0 \
  -o build/custom_prof  ./build/custom_gemm  1024 1024 1024 30
```

### 5.1 kernel 时长对比（HIPOPS 段）

| 实现 | kernel 名 | calls | 单次耗时 | 总耗时 |
|---|---|---|---|---|
| hipBLAS | `Cijk_Ailk_Bljk_SB_MT128x64x16_SE_AMAS3_...` | 33 | **57.7 µs** | 1.91 ms |
| 手写 | `sgemm_tiled(int, int, int, float const*, ...)` | 33 | **696.8 µs** | 23.0 ms |

单 kernel 差距 ~12x，跟上一步计时给的 3.1 TFLOPs vs 37 TFLOPs 大致对得上。差距主要来自寄存器/共享内存复用还没吃满 + 没上向量化 load + 单 buffer。

### 5.2 API 端观察

hipBLAS 版本还额外看到：

- `hipModuleGetFunction` 69 次共 17.5%、`hipModuleUnload` 97 次共 9.1%：rocBLAS/Tensile 会按 shape 加载最合适的 kernel 变体，第一次跑会触发编译/加载。
- `hipMalloc` 6 次占 3.7%：一次性分配，正常。

手写版：

- `hipEventSynchronize` 占 39%、`hipMalloc` 占 50%：kernel 本身相对短所以看起来占比"没那么高"，其实是被 host 侧等待放大了。这是分析 CPU 端瓶颈的常见指标。
- `hipLaunchKernel` 33 次共 552 µs，每次 launch 17 µs，是 kernel 本身耗时的 2.4%，可忽略。

### 5.3 想更细看要怎么办

如果要看 kernel 内部各阶段的时间（load / compute / store 占比、寄存器 spill、共享内存 bank conflict），需要走 `rocprof` + `--kernel-metrics`（DTK 里对应 `hipprof --kernel-stack` 或 `rocprofv2`）。json/perfetto trace 也可以直接拖到 chrome://tracing 或 perfetto UI 里看时间线，判断是不是有 kernel 排队 gap。

## 六、跑法一览

```bash
cd /workspace/new_grad_training/hw6_hipblas_gemm
rm -rf build && mkdir build && cd build
cmake ..                                          # 生成 Makefile
make -j                                           # 编两个 exe
make run                                          # 直接跑对比
make profile                                      # hipprof 采集 trace
./hipblas_gemm 2048 2048 2048 20                  # 换 shape 再跑
./custom_gemm  2048 2048 2048 20
```

## 七、几个复盘

- **对答案是硬要求**：手写 GEMM 很容易索引越界或者 tile 边界写错。小 shape 拉 CPU 参考实现做 `max abs err` 检查，比后期用 profile 反推容易十倍。
- **计时用 GPU event，不要用 chrono**：wall clock 会把 launch 排队和 host 等待也算进去，误导。
- **hipBLAS 是被高度调优过的**：不要指望手写第一版就打过 vendor 库。这个 demo 的意义是理解 tiling / 共享内存 / 寄存器分块的思路，以及 profile 时能读懂 `Cijk_*` 系列 kernel 名字。
- **cmake 值得写清楚**：一个 `make profile` 目标能省很多重复敲命令的时间。CI 里加上一层 `ctest` 甚至能自动回归性能。
