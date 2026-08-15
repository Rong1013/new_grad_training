"""对比 eager 与 torch.compile 的性能，同时观察算子融合行为。

跑法：
    python bench_eager_vs_compile.py
    python bench_eager_vs_compile.py --shape 4096 4096 --iters 200

脚本做两件事：
1. 用一个 elementwise-heavy 的小模型（多个 pointwise + LayerNorm + MLP），
   分别在 eager 和 torch.compile 下测吞吐/延迟。elementwise 多的场景是
   Inductor 融合收益最明显的地方，所以对比更有意义。
2. 用 TORCH_LOGS 打印出 compile 生成的融合 kernel 数量，作为“融合成什么样”
   的直接证据。
"""

import argparse
import os
import statistics
import time
from contextlib import contextmanager

import torch
import torch.nn as nn


class ElementwiseHeavyBlock(nn.Module):
    """一段 pointwise chain + LayerNorm + Linear。eager 下会拆成很多 kernel。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 一串 pointwise，故意写成典型可融合形状
        y = torch.tanh(x) * torch.sigmoid(x) + x * 0.5
        y = y * self.scale + self.bias
        y = self.norm(y)
        y = self.fc1(y)
        y = torch.nn.functional.gelu(y)
        y = self.fc2(y)
        # 再来一段 pointwise 残差
        y = y + x
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
        return y


class ToyModel(nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ElementwiseHeavyBlock(dim) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


@contextmanager
def cuda_timer():
    """用 cuda event 精确测 GPU wall-clock。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    yield lambda: (torch.cuda.synchronize(), start.elapsed_time(end))[1]
    end.record()


def bench(model: nn.Module, x: torch.Tensor, iters: int, warmup: int) -> dict:
    # warmup：compile 版本第一次会触发编译，一定要热身
    for _ in range(warmup):
        y = model(x)
        y.sum().backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    fwd_times, full_times = [], []
    for _ in range(iters):
        # forward only
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x)
        torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000)

        # forward + backward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x)
        y.sum().backward()
        torch.cuda.synchronize()
        full_times.append((time.perf_counter() - t0) * 1000)
        model.zero_grad(set_to_none=True)

    def stat(v):
        return {
            "mean_ms": statistics.mean(v),
            "p50_ms": statistics.median(v),
            "p90_ms": statistics.quantiles(v, n=10)[-1] if len(v) >= 10 else max(v),
            "min_ms": min(v),
        }

    return {"forward": stat(fwd_times), "fwd+bwd": stat(full_times)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs=2, type=int, default=[2048, 1024],
                        help="input shape: batch, dim")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mode", choices=["reduce-overhead", "default", "max-autotune"],
                        default="reduce-overhead")
    parser.add_argument("--dump-fusion", action="store_true",
                        help="打印 Inductor 生成的融合 kernel 数量与文件路径")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 环境"
    device = torch.device("cuda")
    torch.manual_seed(0)

    batch, dim = args.shape
    x = torch.randn(batch, dim, device=device, requires_grad=False)

    print(f"config: shape={tuple(x.shape)} depth={args.depth} "
          f"iters={args.iters} warmup={args.warmup} mode={args.mode}")

    eager_model = ToyModel(dim, args.depth).to(device)
    # 用 deepcopy 保证两份权重一致
    import copy
    compile_target = copy.deepcopy(eager_model)
    compiled_model = torch.compile(compile_target, mode=args.mode, fullgraph=False)

    if args.dump_fusion:
        os.environ.setdefault("TORCH_LOGS", "output_code,graph_breaks,fusion")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "./_inductor_cache")

    print("\n== eager ==")
    r_eager = bench(eager_model, x, args.iters, args.warmup)
    for k, v in r_eager.items():
        print(f"  {k}: {v}")

    print("\n== torch.compile ==")
    r_comp = bench(compiled_model, x, args.iters, args.warmup)
    for k, v in r_comp.items():
        print(f"  {k}: {v}")

    print("\n== speedup ==")
    for k in r_eager:
        e = r_eager[k]["mean_ms"]
        c = r_comp[k]["mean_ms"]
        print(f"  {k}: eager {e:.3f} ms vs compiled {c:.3f} ms -> x{e / c:.2f}")

    if args.dump_fusion:
        cache_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        kernels = []
        for root, _, files in os.walk(cache_dir):
            for name in files:
                if name.endswith(".py") and name.startswith("output_code"):
                    kernels.append(os.path.join(root, name))
        print(f"\ninductor 生成的 kernel 文件数：{len(kernels)}")
        for p in kernels[:5]:
            print(f"  {p}")


if __name__ == "__main__":
    main()
