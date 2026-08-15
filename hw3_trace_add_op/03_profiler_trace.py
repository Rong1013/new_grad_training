"""
03_profiler_trace.py

用 torch.profiler 抓一次 torch.add 的完整调用栈，导出 chrome trace。

对应 add 算子链路文档的 §7 + §10：
  - profiler 会记录 Dispatcher 每一层 kernel 的 event，
    包括 aten::add / add_stub / 具体 backend kernel（gpu_kernel、cpu_kernel 等）
  - `record_shapes=True` 会带上输入 shape/dtype 信息
  - `with_stack=True` 记录 Python 调用栈；`experimental_config=_ExperimentalConfig(verbose=True)`
    在部分构建里能拉到 C++ 栈
  - `with_flops=True` 让 profiler 估算 FLOPs

产出：
  * 控制台按 self_cpu_time_total / self_cuda_time_total 排序的 top-20 表
  * add_forward_trace.json / add_backward_trace.json —— chrome://tracing 可视化
  * add_stack.txt —— 每个 top event 的调用栈（如果可用）
"""

import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def maybe_cuda_activity():
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts


def run_add_forward(device: str, size=(1024, 1024), alpha: float = 2.0, iters: int = 5):
    a = torch.randn(*size, device=device, requires_grad=True)
    b = torch.randn(*size, device=device, requires_grad=True)
    outs = []
    for _ in range(iters):
        with record_function("py::torch.add"):
            c = torch.add(a, b, alpha=alpha)
        outs.append(c)
    if device == "cuda":
        torch.cuda.synchronize()
    return a, b, outs


def run_add_backward(a, b, outs):
    for c in outs:
        with record_function("py::c.sum().backward()"):
            c.sum().backward(retain_graph=True)
    if a.device.type == "cuda":
        torch.cuda.synchronize()


def profile_forward(device: str) -> None:
    tag = f"forward/{device}"
    print("=" * 90)
    print(f"[profiler] {tag}")
    print("=" * 90)
    with profile(
        activities=maybe_cuda_activity(),
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        run_add_forward(device)

    sort_by = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by=sort_by, row_limit=20))

    trace_path = os.path.join(OUT_DIR, f"add_forward_trace_{device}.json")
    prof.export_chrome_trace(trace_path)
    print(f"  -> chrome trace saved: {trace_path}\n")


def profile_backward(device: str) -> None:
    tag = f"backward/{device}"
    print("=" * 90)
    print(f"[profiler] {tag}")
    print("=" * 90)
    a, b, outs = run_add_forward(device, iters=3)
    with profile(
        activities=maybe_cuda_activity(),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        run_add_backward(a, b, outs)

    sort_by = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=25))

    trace_path = os.path.join(OUT_DIR, f"add_backward_trace_{device}.json")
    prof.export_chrome_trace(trace_path)
    print(f"  -> chrome trace saved: {trace_path}\n")


def dump_add_events(device: str) -> None:
    """把 profile 里所有 name 含 add 的原始事件 dump 出来，看 Dispatcher 每一层。"""
    print("=" * 90)
    print(f"[raw events with 'add' in name]  device={device}")
    print("=" * 90)
    with profile(
        activities=maybe_cuda_activity(),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        run_add_forward(device, iters=1)

    events = prof.events()
    hits = [e for e in events if "add" in e.name.lower()]
    # 只留每个 name 的前几条，避免刷屏
    seen: dict[str, int] = {}
    for e in hits:
        seen[e.name] = seen.get(e.name, 0) + 1
        if seen[e.name] > 2:
            continue
        cpu_us = e.cpu_time_total
        cuda_us = getattr(e, "cuda_time_total", 0)
        shapes = getattr(e, "input_shapes", None)
        print(f"  {e.name:60s} cpu={cpu_us:8.1f}us  cuda={cuda_us:8.1f}us  shapes={shapes}")
    print()


if __name__ == "__main__":
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print()

    profile_forward("cpu")
    profile_backward("cpu")
    dump_add_events("cpu")

    if torch.cuda.is_available():
        # warm-up 一次，让 cudnn/cublas 初始化不进 profile
        _ = torch.randn(4, 4, device="cuda") + 1
        torch.cuda.synchronize()
        profile_forward("cuda")
        profile_backward("cuda")
        dump_add_events("cuda")
    else:
        print("[note] cuda not available, skip cuda profile")
