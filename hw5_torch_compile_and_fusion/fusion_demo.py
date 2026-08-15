"""对照实验：哪些算子融合了、哪些没融合。

跑法：
    TORCH_LOGS=output_code python fusion_demo.py

跑完在标准错误里能看到 Inductor 生成了几个 triton kernel、每个 kernel 的
名字里带了哪些 op。名字里的 `fused_A_B_C` 就是这次融合覆盖到的算子。
"""

import torch
import torch.nn as nn


def case_pointwise_chain(x: torch.Tensor) -> torch.Tensor:
    """典型能融的一串 pointwise。Inductor 会合并成 1 个 kernel。"""
    y = torch.sin(x)
    y = y * 2.0
    y = torch.tanh(y)
    y = y + x
    y = torch.nn.functional.gelu(y)
    return y


def case_pointwise_plus_reduce(x: torch.Tensor) -> torch.Tensor:
    """pointwise -> reduce(mean) -> pointwise，能融成 1 个 persistent-reduction kernel。"""
    y = torch.tanh(x) * torch.sigmoid(x)
    m = y.mean(-1, keepdim=True)
    return (y - m) * torch.rsqrt(y.var(-1, keepdim=True) + 1e-6)


def case_matmul_then_pointwise(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """matmul 后接 pointwise。GEMM 是独立 kernel，后面的 bias/gelu 才能融进去。"""
    y = x @ w1 + b1
    y = torch.nn.functional.gelu(y)
    y = y * 1.5
    return y


def case_two_matmuls(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """两个连续 GEMM 之间不会融合。会看到 2 个 mm/addmm kernel。"""
    y = x @ w1
    y = y @ w2
    return y


def case_data_dependent_break(x: torch.Tensor) -> torch.Tensor:
    """出现依赖 GPU 值的 Python 分支，会打断 graph 或者退化。"""
    y = torch.tanh(x)
    if bool(y.sum() > 0):
        y = y * 2
    else:
        y = y + 1
    return y


def case_reduction_then_reduction(x: torch.Tensor) -> torch.Tensor:
    """维度不同的两次 reduce，融合会退化 —— 一般是两个 kernel。"""
    y = x.sum(-1)
    z = y.sum(-1)
    return z


def main() -> None:
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    x = torch.randn(2048, 1024, device=dev)

    print("== case1: pure pointwise chain (期望 1 个融合 kernel) ==")
    fn1 = torch.compile(case_pointwise_chain)
    _ = fn1(x); torch.cuda.synchronize()

    print("\n== case2: pointwise + reduce + pointwise (期望 1 个 persistent-reduction kernel) ==")
    fn2 = torch.compile(case_pointwise_plus_reduce)
    _ = fn2(x); torch.cuda.synchronize()

    print("\n== case3: matmul + pointwise (期望 GEMM + pointwise 至少 2 个 kernel，bias/gelu 会融进 pointwise) ==")
    w1 = torch.randn(1024, 1024, device=dev)
    b1 = torch.randn(1024, device=dev)
    fn3 = torch.compile(case_matmul_then_pointwise)
    _ = fn3(x, w1, b1); torch.cuda.synchronize()

    print("\n== case4: matmul → matmul (期望 2 个独立 GEMM，无融合) ==")
    w2 = torch.randn(1024, 512, device=dev)
    fn4 = torch.compile(case_two_matmuls)
    _ = fn4(x, w1, w2); torch.cuda.synchronize()

    print("\n== case5: data-dependent branch (期望 graph break) ==")
    fn5 = torch.compile(case_data_dependent_break)
    _ = fn5(x); torch.cuda.synchronize()

    print("\n== case6: reduce → reduce 跨轴 (期望 2 个 reduce kernel) ==")
    x2 = torch.randn(64, 128, 256, device=dev)
    fn6 = torch.compile(case_reduction_then_reduction)
    _ = fn6(x2); torch.cuda.synchronize()


if __name__ == "__main__":
    main()
