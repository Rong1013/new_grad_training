"""
目标：
    1. 用 PyTorch 调用矩阵乘法 (torch.matmul / @ / nn.Linear)
    2. 使用 torch.profiler 观察底层调用（CPU op、CUDA/HIP kernel、内存搬运）
    3. 导出 chrome tracing 文件，可在 chrome://tracing 或 https://ui.perfetto.dev 中可视化

关注点：
    - PyTorch 上层的 aten::matmul / aten::mm / aten::addmm 会被分发到什么后端 kernel
    - GEMM kernel（cutlass / rocblas / cublasLt）在 GPU 上的耗时占比
    - Host→Device / Device→Host 拷贝时间
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule


def run_matmul_workload(device: torch.device):
    """
    准备一组不同规模的矩阵乘法，用于观察不同 shape 下 kernel 选择的差异。
    典型 GEMM 尺寸 (M, K) x (K, N) -> (M, N)
    """
    shapes = [
        (512, 512, 512),      # 小规模：可能触发小 kernel / 甚至 CPU fallback
        (2048, 2048, 2048),   # 中规模：典型的 GEMM
        (4096, 4096, 4096),   # 大规模：充分暴露 GPU 计算峰值
    ]

    for (M, K, N) in shapes:
        # record_function 会在 profiler 中显示一个自定义标签块，方便定位
        with record_function(f"matmul_{M}x{K}x{N}"):
            # 每次都新建，避免复用缓存混淆时间
            a = torch.randn(M, K, device=device, dtype=torch.float32)
            b = torch.randn(K, N, device=device, dtype=torch.float32)

            # 三种等价写法都会最终走到 aten::mm / aten::matmul
            c1 = torch.matmul(a, b)
            c2 = a @ b
            # nn.Linear 内部会走 aten::addmm（带 bias 的 GEMM）
            linear = torch.nn.Linear(K, N, bias=True).to(device)
            c3 = linear(a)

        # 强制同步，保证下一轮开始前当前 GPU work 已完成
        # 否则 profiler 记录的 kernel 时间会跨到下一个 record_function 中
        if device.type in ("cuda", "hip"):
            torch.cuda.synchronize()

        # 防止编译器优化掉未使用的输出
        del c1, c2, c3


def main():
    # 优先使用 GPU（CUDA 或 ROCm/HIP 均通过 torch.cuda 接口访问）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")

    # 需要 profile 的 activity：
    #   - CPU: 记录 Python / aten 层 op
    #   - CUDA: 记录 GPU kernel、memcpy（HIP 也复用这个 flag）
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # warmup：先跑一次让 cuBLAS / rocBLAS 完成 kernel 选择（heuristic tuning）
    # 否则第一次调用的耗时里混着 tuning 开销，会污染观测结果
    print("[info] warmup...")
    run_matmul_workload(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # schedule 参数含义：
    #   wait=1     跳过前 1 步（冷启动）
    #   warmup=1   再跑 1 步用于 profiler 自身预热
    #   active=2   正式记录 2 步
    #   repeat=1   整个循环重复 1 次
    prof_schedule = schedule(wait=1, warmup=1, active=2, repeat=1)

    trace_path = "matmul_trace.json"
    print(f"[info] profiling... trace will be saved to {trace_path}")

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,       # 记录 tensor shape，便于定位是哪个 GEMM
        profile_memory=True,      # 记录内存分配/释放
        with_stack=False,         # 关闭 python stack，减少开销
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    ) as prof:
        # schedule 要求外层用循环驱动 step()，每次 step 计入一个 iteration
        for step in range(4):
            run_matmul_workload(device)
            prof.step()

    # 打印 CPU/GPU 各 top op 的耗时排序
    # sort_by="cuda_time_total" 在 CPU-only 环境下会 fallback，无需 if 判断
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print("\n===== Top 20 ops =====")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    main()
