# 作业 2：PyTorch 调用矩阵乘法 / LayerNorm 并用 profiler 观察底层调用

## 目录内容

| 文件 | 说明 |
|---|---|
| `matmul_profile.py` | 用 `torch.matmul` / `@` / `nn.Linear` 跑三种 shape 的 GEMM，导出 chrome trace |
| `layernorm_profile.py` | 用 `nn.LayerNorm` 跑三种 shape，分别 profile forward-only 与 forward+backward |

## 运行

```bash

# 任务 1：矩阵乘法
python matmul_profile.py

# 任务 2：LayerNorm
python layernorm_profile.py
```

## 查看 trace

生成的 `*.json` 是 Chrome Tracing 格式，可以：

1. 打开 Chrome 浏览器，访问 `chrome://tracing`，Load 该 json 文件；
2. 或访问 <https://ui.perfetto.dev>，Open trace file，加载 json。

在 Perfetto 里可以看到：
- **CPU 侧**：Python 调用 → `aten::matmul` / `aten::native_layer_norm` 的分发链路；
- **GPU 侧**：实际执行的 kernel 名称（如 `ampere_sgemm_*` / `vectorized_layer_norm_kernel`）；
- **memcpy**：Host↔Device 数据搬运的耗时；
- **shape 信息**：`record_shapes=True` 让每个 op 都带上 tensor shape。

## profiler 输出解读要点

- `Self CUDA` 是 kernel 本身的执行时间，不包括调用它的上层 op；
- `CUDA total` 是包含子 op 的总时间；
- `# of Calls` 显示同名 op 被调用次数；
- `record_function("xxx")` 出现的自定义 label 便于定位是哪个 shape/阶段的耗时。

## 关注问题

1. `nn.Linear` 与 `torch.matmul` 底层 kernel 有何不同？（提示：`aten::addmm` vs `aten::mm`）
2. LayerNorm 反向为什么会调用 reduce kernel？（gamma/beta 的梯度需要跨 batch 求和）
3. 小矩阵（512x512）为什么 kernel launch 开销占比高？
