#  模型性能分析笔记

## 一、`train_perf_bottleneck_model.py` 是干什么的

这是一份故意写慢的 PyTorch 训练脚本，用来练手 profiler。里面塞了一堆常见的坑，用 `torch.profiler.record_function` 标好了每一段的名字，跑起来能在 chrome trace 里直接对上号。

主要几块：

- `SlowPatternDataset`：合成图数据集。`__getitem__` 里搞了几件坏事：每次调用都新建一个 `torch.Generator` 并 `manual_seed(index)`；然后跑几轮 `image.flip(-1).contiguous().flip(-1)`，正反两次 flip 其实等于什么都没做，但每次都物化了新张量；最后还 `time.sleep(data_sleep_ms)` 假装 IO 慢。
- `BottleneckNet`：小 CNN + MLP。`forward` 里三段被点名的：
  - `bottleneck_single_large_matmul`：手写一个大 `Parameter` 做 `x @ W`。
  - `bottleneck_fragmented_small_ops`：用 `torch.chunk` 拆 16 份，Python for 循环里给每份跑 relu/sigmoid/mul/clone，最后 cat 回来。这是典型的“把一次向量化拆成一堆小 kernel”反模式。
  - `bottleneck_frequent_allocations`：`empty_like`、`clone`、`transpose().contiguous().transpose().contiguous()`（相当于什么也没干）循环若干次，纯粹在制造内存分配。
- `run_epoch`：训练循环里额外插了几段：
  - `bottleneck_h2d_copy_batch`：`.to(device)`。
  - `bottleneck_cpu_python_work`：`for i in range(3000): checksum += (i*i) % 97`，纯 Python 空转。
  - `bottleneck_extra_memcpy_and_sync`：在 CPU 上 `torch.randn` 出小噪声，然后 `.to(device)` 加到 images 上。
  - `bottleneck_loss_temp_allocations`：手写 one-hot + softmax + pow + mean，塞给 loss。
  - `bottleneck_d2h_sync_for_logging`：每个 step 都 `.cpu()` + `.item()` 取 loss/acc 打日志。
- DataLoader 参数：`num_workers=0`，因为 workers 为 0，脚本里的 `pin_memory` 判断也就顺带关掉了。

## 二、profiler 数据说了什么

看 `output.log` 和 `profiler_traces/bottleneck_trace_cuda.json`，几个数字比较扎眼：

| 指标 | 数值 |
|---|---|
| 训练总耗时 | 7.87 s |
| Self CPU 总时间 | 7.155 s |
| Self CUDA 总时间 | 22.9 ms |
| `aten::miopen_convolution` self CPU | 4.27 s，占 59.7% |
| `aten::addmm` self CPU | 472 ms |
| `bottleneck_fragmented_small_ops` CUDA | 37.0 ms（GPU 侧最慢的一段） |
| `bottleneck_loss_temp_allocations` CUDA | 25.7 ms |
| `bottleneck_extra_memcpy_and_sync` CUDA | 18.3 ms |
| `bottleneck_d2h_sync_for_logging` CUDA | 11.1 ms |
| 主线程 tid=83261 单独扛了 63k 条 event，没有 dataloader worker 线程 |

结论很直白：整个训练是 CPU-bound。CPU 花了 7 秒多，GPU 只干了 20 多毫秒的活，剩下时间 GPU 基本在等。等谁？主要等三件事：DataLoader 单进程阻塞主线程；每个 step 一堆 `.item()` 强制同步；主线程还被无用 Python 循环和 CPU 上的 randn 挤着。

## 三、这份脚本的三处优化点

### 优化点 1：数据加载 —— 开 worker 进程 + 删掉没用的 CPU 预处理

对应位置：`SlowPatternDataset.__getitem__`（第 47-62 行）和 DataLoader 构造（第 278-284 行）。

`num_workers=0` 直接导致 `pin_memory` 也是 `False`。trace 里 `aten::flip` 出现了 2048 次、`aten::to` 3801 次，全部落在主线程，GPU 只能干等。`flip(-1).contiguous().flip(-1)` 这段本身语义上就是 no-op，还有 `time.sleep(0.2ms)` × 256 sample × 2 epoch ≈ 100 ms 纯等待，都是白花。

改：

- DataLoader 加 `num_workers=4, pin_memory=True, persistent_workers=True`。
- 把 `flip(-1).contiguous().flip(-1)` 和 `time.sleep` 都删掉。
- `torch.Generator().manual_seed(index)` 如果只是给噪声用种子，其实没必要每次新建；要么外面复用，要么直接用全局 RNG。

### 优化点 2：拿掉每 step 的 D2H 同步和多余的 CPU→GPU 拷贝

对应位置：

- 第 197-199 行 `bottleneck_extra_memcpy_and_sync`，CPU 上生成噪声再 `.to(device)`。
- 第 215-219 行 `bottleneck_d2h_sync_for_logging`，`logits.cpu()`、`labels.cpu()`、`.item()`，等于每个 step 都 `hipDeviceSynchronize` 一次。
- 第 192-195 行 `bottleneck_cpu_python_work`，纯 Python 3000 次循环算个 checksum，跟训练完全无关。

trace 里 `hipMemcpyWithStream` 96 次共 11.4 ms，`bottleneck_d2h_sync_for_logging` GPU 侧 11.1 ms，每次 `.item()` 都会让 GPU 排队被 flush，这也是 `aten::miopen_convolution` self CPU 高达 4.27 s 的一个原因 —— 每次同步之后 CPU 都要重新排队 kernel。

改：

- 噪声直接在 GPU 上生成：`torch.randn(..., device=device)`，别在 CPU 生成再拷过去。
- loss/acc 的累加留在 GPU tensor 上（`running_loss += loss.detach() * bs`；`running_correct += (logits.argmax(1)==labels).sum()`），epoch 末尾再 `.item()` 一次。
- `bottleneck_cpu_python_work` 那段直接删。

### 优化点 3：`fragmented_small_ops` 向量化 + 删死代码 + loss 用官方接口

对应位置：

- 第 100-108 行 `bottleneck_fragmented_small_ops`：16 次 chunk，每次输出只有 `[32,16]`，kernel 本身几微秒，launch 开销把它盖过去。GPU 上这一段合计 37 ms，是所有 record_function 段里最贵的。
- 第 110-117 行 `bottleneck_frequent_allocations`：整段其实是死代码，两次 `transpose().contiguous()` 完全恢复原状，最后加回来的量级是 1e-4 和 1e-5，对训练几乎无影响。trace 里 `aten::empty` 1217 次、`aten::empty_like` 2194 次，主要来源就是这段。
- 第 205-210 行 `bottleneck_loss_temp_allocations`：手写 one-hot + softmax + pow + mean，等于把 label-smoothing 的功能又慢又碎地实现了一遍。

改：

- fragmented 那段用一次向量化替代。想给每个 chunk 加不同 bias，就构造一个长向量 `bias = torch.arange(1, K+1, device=x.device) * 1e-4` 再 `repeat_interleave(chunk_size)` 加到 `x` 上，然后一次 `relu`、一次 `sigmoid*` 完事，不需要 chunk/cat/clone。
- `bottleneck_frequent_allocations` 整段删。
- 用 `nn.CrossEntropyLoss(label_smoothing=0.02)`，或者直接不做额外 loss 项。

按这三点改下来，7.87 s 的 wall clock 预计能压到 1-2 s 量级。

## 四、BERT4Rec profile 的看法

BERT4Rec 那份 `BERT4Rec.pt.trace.json` 记了 5 个 ProfilerStep，总窗口大约 472 ms。分类统计：

| 类别 | 总时长 (µs) |
|---|---|
| CPU ops | 915 146 |
| GPU kernels | 193 562 |
| GPU memcpy | 20 907 |
| `hipMemcpyWithStream` | 163 209（1930 次）|
| `aten::item` | 166 270（2590 次，平均 518 次/step）|
| `aten::isnan` + `any` + `_is_any_true` + `ne` | 约 65 000（1850+ 次）|
| Optimizer.step | 12 463 |
| DataLoader `_SingleProcessDataLoaderIter` | 41 814（5 次，8.4 ms/step）|

真正跑数的 GEMM (`Cijk_*` 系列) 加起来只有约 20 ms，不到窗口的 5%，说明矩阵乘、attention、优化器都不是瓶颈。这份 profile 里问题出在两个地方。

### 建议 1：干掉每 step 518 次的 `.item()`

`aten::item` 每个 step 稳定出现 518 次，输入类型都是 bool 标量，同时能看到 `aten::ne / isnan / any / _is_any_true` 各出现 1850 次左右。这是训练循环里对每个参数或中间张量都做 `bool(torch.isnan(x).any())` 或 `assert not torch.any(...).item()` 的典型指纹，很可能是打开了 `torch.autograd.set_detect_anomaly(True)`，或者手写的调试断言。

每次 `.item()` 都会触发一次 `hipMemcpyWithStream`，语义上就是 device → host 的同步等待，直接把 CPU/GPU 流水打断。这两项加起来 166 ms + 163 ms ≈ 329 ms，占 472 ms 窗口的约 70%。

怎么改：

- 生产训练不要开 anomaly detection。
- 如果确实要检查 NaN，把结果留在 GPU 累加，比如 `nan_flag |= torch.isnan(loss)`，只在 `step % N == 0` 时 `.item()` 一次。
- AMP 场景下用 `GradScaler` 自带的 skip-step 逻辑更合适，它的判据是异步的。

只把这一项拿掉，每个 step 从 82 ms 大概能落到 30 ms 附近，GPU 占用率能从 47% 拉到 90% 以上。

### 建议 2：DataLoader 多进程 + `pin_memory` + `non_blocking` 拷贝

trace 里 `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` 明确写着单进程，5 次共 41.8 ms，平均 8.4 ms/step，占 step 时间的 10%。这段时间 GPU 是空的，主线程在等数据。

怎么改：

```python
DataLoader(
    dataset,
    batch_size=...,
    num_workers=4,            # 或者物理核数 - 1
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

# 训练循环里
batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
```

配合建议 1 把同步去掉之后，`non_blocking=True` 的 H2D 拷贝就能真正跟上一个 step 的 GPU 计算重叠，8.4 ms/step 的等待基本能藏起来。

## 五、几个自己回头要记住的点

- CPU self time 远大于 GPU self time 时，别去优化 kernel，去看是不是同步/DataLoader/纯 Python 挤了主线程。
- `.item()` / `.cpu()` / `print(loss)` 在训练循环里非常贵，一定要挪到统计窗口末尾。
- 一堆 `flip / transpose().contiguous() / empty_like` 之类看着无害的操作，累积起来能吃掉半个 profile。
- chunk 循环出小 kernel 是很典型的坑，能向量化就向量化，不能就想办法用 `bmm` / `einsum` 合成大 kernel。
- 看 profile 先看类别 (cpu_op / kernel / memcpy / runtime) 分布，再看具体算子。类别分布告诉你“瓶颈在哪个环节”，算子清单告诉你“具体改哪一行”。
