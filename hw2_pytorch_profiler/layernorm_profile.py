"""
目标：
    1. 用 PyTorch 调用 nn.LayerNorm / F.layer_norm
    2. 用 torch.profiler 观察其底层 kernel
    3. 对比 forward-only 与 forward+backward 两种场景下的 kernel 差异

关注点：
    - aten::native_layer_norm 会分发到什么 kernel（vectorized / persistent 等）
    - LayerNorm 反向 (aten::native_layer_norm_backward) 会涉及多少 reduce kernel
    - shape 极端时（比如 hidden_size 很大）kernel 是否会切换实现
"""

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, schedule


def run_layernorm_workload(device: torch.device, train: bool):
    """
    构造几种典型 shape 的 LayerNorm。
    Transformer 中 LayerNorm 常见输入 shape：(batch, seq_len, hidden)
    normalized_shape 通常是最后一维 hidden。
    """
    configs = [
        # (batch, seq_len, hidden)
        (32, 128, 768),    # BERT-base
        (16, 512, 1024),   # BERT-large 中等序列
        (8, 2048, 4096),   # 大模型场景，hidden 很大
    ]

    for (B, S, H) in configs:
        with record_function(f"layernorm_{B}x{S}x{H}"):
            # 输入张量；训练时需要 requires_grad 以触发反向
            x = torch.randn(B, S, H, device=device, dtype=torch.float32,
                            requires_grad=train)

            # nn.LayerNorm 内部封装了可学习的 gamma/beta，等价于 F.layer_norm(x, (H,), w, b)
            ln = torch.nn.LayerNorm(H).to(device)

            # forward
            y = ln(x)

            if train:
                # 用一个简单的 loss 触发反向：sum() 是最轻量的标量化操作
                loss = y.sum()
                loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # 释放引用，避免下一轮 shape 记录混淆
        del x, y, ln


def profile_one_mode(device: torch.device, train: bool, trace_name: str):
    """
    针对一种模式（是否 train）跑一次 profiler，导出独立的 trace。
    分开两个 trace 更容易在时间轴上对比 forward-only vs forward+backward。
    """
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # warmup 让 cuDNN/rocDNN 的 kernel 选择稳定下来
    print(f"[info] warmup ({'train' if train else 'eval'})...")
    run_layernorm_workload(device, train)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"[info] profiling {'train' if train else 'eval'}, trace -> {trace_name}")
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=lambda p: p.export_chrome_trace(trace_name),
    ) as prof:
        for _ in range(4):
            run_layernorm_workload(device, train)
            prof.step()

    # 打印聚合表格。关注 aten::native_layer_norm(_backward) 及其分发到的 kernel
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(f"\n===== Top 20 ops ({'train' if train else 'eval'}) =====")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")

    # 1) forward-only：只看正向 kernel
    profile_one_mode(device, train=False, trace_name="layernorm_forward_trace.json")

    # 2) forward + backward：观察反向 kernel（native_layer_norm_backward
    #    通常会拆成多个 elementwise + reduce kernel）
    profile_one_mode(device, train=True, trace_name="layernorm_train_trace.json")


if __name__ == "__main__":
    main()
