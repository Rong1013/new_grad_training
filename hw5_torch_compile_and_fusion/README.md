# HW5: torch.compile vs eager，以及算子融合能干什么、不能干什么

这份笔记回答两件事：

1. eager 执行和 `torch.compile` 执行差在哪里，为什么后者会更快。
2. 一个模型里，哪些算子之间能融、哪些不能融，融合到底能省什么。

代码分两个：

- `bench_eager_vs_compile.py`：用一个 pointwise + LayerNorm + Linear 的小
  block 测吞吐，对比 eager 和 `torch.compile`。
- `fusion_demo.py`：几个对照 case，用 `TORCH_LOGS=output_code` 把 Inductor
  实际生成的 triton kernel 名字打印出来，直接看融合到了哪一步。

## 一、eager vs torch.compile 的性能差异

### 1.1 实测数字（同一台机，ROCm gfx936）

`bench_eager_vs_compile.py --shape 2048 1024 --depth 4 --iters 100 --warmup 30 --mode default`

只看 min 时间（避开 warmup 尾巴和偶发抖动）：

| 场景 | eager | compiled | 加速比 |
|---|---|---|---|
| forward only | 3.12 ms | 2.39 ms | x1.31 |
| forward + backward | 8.64 ms | 6.38 ms | x1.35 |

`reduce-overhead` 模式在这个 shape 上因为要走 CUDA graph 记录 + 重录制，
mean 会被拖偏，但稳定态下 p50 是 eager 3.18 ms → compiled 3.03 ms，也是快的。
真正跑的时候用 `default` mode 比较更公平。

值得注意的是：如果模型本身几乎全是大 GEMM（比如 batch 特别大的纯 MLP），
compile 的加速就会很小，因为 GEMM 本来就是 cuBLAS/hipBLAS 里的高度调优内核，
compile 没什么可以再省的。**compile 的收益跟你模型里 pointwise / reduce
的占比高度相关**，这也是我们 demo 选 elementwise-heavy 结构的原因。

### 1.2 为什么会更快

eager 每一次 op 调用都要走一遍：

- Python 层 dispatch（Autograd → CPU/CUDA → 具体 backend）
- 每个 op 单独启动一个 CUDA kernel
- 每个 kernel 输入输出都要走一遍 global memory 读写

`torch.compile` 走的是 TorchDynamo（前端抓 fx graph）+ AOTAutograd（前后向一起
抓）+ Inductor（把 fx graph 下沉成 Triton kernel）。它带来的加速主要来自四件事：

1. **算子融合（最大头）**：连续的 pointwise/reduce 被合成一个 kernel，中间
   张量只留在寄存器/shared memory，不落回 HBM。省的是内存带宽和 kernel launch
   开销。elementwise 场景通常 memory-bound，融合直接砍掉大部分 HBM 流量。
2. **kernel launch 次数减少**：eager 一层 block 可能派发 10+ 个 kernel，
   compile 之后合并成 2–3 个。每次 launch 大概几微秒，对小 batch 尤其重要。
3. **Python overhead 消除**：编译后走 C++ 侧的调度，没有 Python dispatcher、
   没有 autograd Python 代码路径。
4. **CUDA graph（reduce-overhead 模式）**：把整段 kernel 序列录成一个 graph，
   一次 launch 完成，进一步压 launch 开销。代价是形状必须稳定。

### 1.3 实证：Inductor 到底融了什么

跑 `bench_eager_vs_compile.py` 时加上 `TORCH_LOGS=output_code`，可以看到
Inductor 为我们那个 block 生成的 kernel 名字：

```
triton_per_fused_add_mul_native_layer_norm_sigmoid_tanh_0
triton_poi_fused_gelu_1
triton_poi_fused_add_addmm_2
```

翻译一下：`tanh + sigmoid + mul + add + layer_norm` 被打包进了同一个 kernel
（`per_fused_...` 是 Inductor 的持久 reduction kernel 命名），`gelu` 单独一个
pointwise，`addmm + add` 融了一起。原本 eager 下的十几个小 op，编译后压成了
三个 kernel + 两个 matmul（`matmul` 会走 hipBLAS，不参与 triton 融合）。

## 二、哪些算子之间能融，哪些不能

这部分靠 `fusion_demo.py` 的 6 个对照 case 说明。跑命令：

```bash
TORCH_LOGS="output_code" python3 fusion_demo.py 2>&1 \
  | grep -oE "triton_(poi|per|red)_fused[_a-z0-9]*" | sort -u
```

实际输出（同一台 gfx936）：

```
triton_per_fused_add_mean_mul_rsqrt_sigmoid_sub_tanh_var_0   # case2
triton_per_fused_gt_sum_1                                     # case5 (graph break 后半段)
triton_per_fused_sum_0                                        # case6 第一次 reduce
triton_per_fused_sum_1                                        # case6 第二次 reduce
triton_poi_fused_add_0                                        # case5 前半段
triton_poi_fused_add_gelu_mul_0                               # case3 pointwise 部分
triton_poi_fused_add_gelu_mul_sin_tanh_0                      # case1
triton_red_fused_sum_tanh_0                                   # case6 tanh+sum 融了
```

### 2.1 能融的三类

**（1）连续的 elementwise / pointwise 链**
   - 例：`sin -> mul -> tanh -> add -> gelu`（case1）
   - 结果：`triton_poi_fused_add_gelu_mul_sin_tanh_0`，一个 kernel 全搞定。
   - 原理：每个元素独立算，不需要跨 element 的数据交换，天然可以在一个
     kernel 里连着算，中间值全留寄存器。

**（2）pointwise + 简单 reduce（且 reduce 轴在最内层，能装进一个 block）**
   - 例：`tanh*sigmoid -> mean -> var -> sub -> rsqrt -> mul`（case2）
   - 结果：`triton_per_fused_add_mean_mul_rsqrt_sigmoid_sub_tanh_var_0`，
     一个 persistent reduction kernel。这是 LayerNorm/RMSNorm 加速的根源。
   - 原理：Inductor 的 persistent reduction 会在 kernel 里先做 reduce，把
     结果留在 shared memory / 寄存器，紧接着的 pointwise 阶段直接用，不用
     写回 HBM 再读一次。

**（3）GEMM 后紧跟的 bias/激活/scale**
   - 例：`x @ w + b -> gelu -> mul`（case3）
   - 结果：GEMM 本身走 hipBLAS 的 `Cijk_*` kernel，后面的 `add + gelu + mul`
     被融成 `triton_poi_fused_add_gelu_mul_0`。
   - 严格说这不是 GEMM 内部 epilogue 融合（那要 `max-autotune` 或者
     `_scaled_mm` 之类），但在 Inductor 层面，紧邻 GEMM 的 pointwise 一定会被
     单独合成一个 kernel，省掉“GEMM 输出 → 单独 bias kernel → 单独 gelu
     kernel”的往返。

### 2.2 不能融（或者不会融）的三类

**（1）两个 GEMM 之间**
   - 例：`(x @ w1) @ w2`（case4）
   - 结果：两个独立的 hipBLAS GEMM kernel，中间张量必须落 HBM 再读回来。
   - 原因：GEMM 是密集矩阵乘，内部 tiling / shared memory 布局是高度定制的，
     跟 pointwise 的融合模型完全不兼容。就算逻辑上是 `A@B@C`，也得先物化
     `A@B` 再算下一次。想避免这种物化只能靠算法层重排（比如 attention 里
     的 flash attention 把 `softmax(QK^T)V` 手写成一个 fused kernel）。

**（2）依赖 GPU 张量数值的 Python 控制流**
   - 例：`if bool(y.sum() > 0): ...`（case5）
   - 结果：Dynamo 触发 graph break，前半段 `tanh+add` 编到
     `triton_poi_fused_add_0`，`sum + gt` 落到 `triton_per_fused_gt_sum_1`，
     后半段的分支重新走 eager。整个 forward 被切成多段，跨段之间不能融。
   - 原因：Dynamo 只能编“形状 & 控制流不依赖运行时 tensor 值”的静态子图。
     `if bool(...)` 依赖 GPU 数据的 host 值，编译器无法在编译期选分支，
     只能 break。要避免就把分支改成 `torch.where` 之类的张量 op。

**（3）跨维度 / 跨轴的 reduce**
   - 例：`x.sum(-1).sum(-1)`（case6，注意这里跨了两个不同的轴）
   - 结果：两个独立 reduce kernel（`triton_per_fused_sum_0` +
     `triton_per_fused_sum_1`）。case6 里 `tanh + sum(-1)` 倒是融进了一个
     `triton_red_fused_sum_tanh_0`，但**下一层 sum 换了轴之后就融不进去了**。
   - 原因：不同的 reduce 轴意味着 kernel 内的线程/块布局完全不同，一个 kernel
     里没法同时高效处理。同轴的多次 reduce 可以合并（比如 mean+var 一起
     算），跨轴的必须分开。

### 2.3 还有一类会“融但可能反而慢”的坑

- **过大的 shared memory / 寄存器压力**：如果一条融合链太长，Triton 会因为
  寄存器溢出（register spill）反而变慢。这时可以看 `TORCH_LOGS=schedule`
  或 `perf_hints`，或者干脆手工在中间加一个 `.contiguous()` 打断融合。
- **视图操作**（`view/reshape/permute`）：Inductor 把它们当元数据变换，
  不会产生 kernel，但会影响后面的融合边界。
- **随机数**：`torch.rand` 之类算子跨调用状态不同，融合边界会被打断。

## 三、几点上手结论

- 想让 `torch.compile` 出成绩：给它 elementwise 密集的模型（比如 Transformer
  的 FFN、norm、bias+激活），或者 batch 特别小、launch overhead 占大头的场景。
- 反过来，如果你的瓶颈是纯大 GEMM 或者是数据加载/同步（参考 hw4 里的
  BERT4Rec 分析），compile 加速有限，别期待过高。
- 写模型时想主动配合融合器：连续 pointwise 别插 `.item()`、别插 CPU 计算、
  别插形状变化的分支；bias/激活/scale 尽量紧挨着写；能用 `torch.where`
  就别用 Python `if`。
- 想直接看融合效果：`TORCH_LOGS=output_code`（看生成代码）或者
  `TORCH_LOGS=+inductor,+schedule`（看调度决策）。生成的 kernel 名字里
  `fused_A_B_C` 就是这一段被合并的算子清单，很好读。

## 四、跑法

```bash
# 性能对比
python3 bench_eager_vs_compile.py --shape 2048 1024 --depth 4 --iters 100 --warmup 30 --mode default

# 融合行为对照
TORCH_LOGS="output_code" python3 fusion_demo.py 2>&1 \
  | grep -oE "triton_(poi|per|red)_fused[_a-z0-9]*" | sort -u
```

想看每个 case 完整的 triton 源码，去掉 `grep` 即可，或者把
`TORCHINDUCTOR_CACHE_DIR=./_inductor_cache` 加上，Inductor 会把生成的
Python + Triton 源码留在这个目录，逐个 kernel 看更清楚。
