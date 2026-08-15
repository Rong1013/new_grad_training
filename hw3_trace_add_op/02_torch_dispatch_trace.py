"""
02_torch_dispatch_trace.py

动态视角：用 TorchDispatchMode 拦截一次 torch.add(...) 的 Dispatcher 调用。

对应 add 算子链路文档的 §10 (端到端调用链)：

    Python torch.add(a, b)
      → THPVariable_add                              (python_torch_functions.cpp)
      → at::_ops::add_Tensor::call                   (Operators_*.cpp)
      → c10::Dispatcher::call                        <-- 我们在这里挂钩
          ├── AutogradXPU: VariableType::add_Tensor
          └── (redispatch) CPU/CUDA: wrapper_*_add_Tensor
              └── structured_ufunc_add_XPU::impl → add_stub → REGISTER_DISPATCH
                  → ufunc::add(a, b, alpha)         (aten/src/ATen/native/ufunc/add.h)

TorchDispatchMode 会在 __torch_dispatch__ 层看到所有走完 Autograd 之后的 op；
Autograd 侧的分派另外用 torch.autograd.graph.saved_tensors_hooks / grad_fn 观察。
"""

import contextlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class TraceDispatchMode(TorchDispatchMode):
    """把每次落到 __torch_dispatch__ 的 op 打印出来，同时执行真实内核。"""

    def __init__(self, tag: str = ""):
        super().__init__()
        self.tag = tag
        self.calls: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # func 是 OpOverload，例如 aten.add.Tensor
        overload = f"{func}"          # 'aten.add.Tensor'
        schema = func._schema         # c10::FunctionSchema
        # 参数摘要
        def brief(x):
            if isinstance(x, torch.Tensor):
                return f"Tensor(dev={x.device}, dtype={x.dtype}, shape={tuple(x.shape)}, req_grad={x.requires_grad})"
            return repr(x)
        arg_str = ", ".join(brief(a) for a in args)
        kw_str = ", ".join(f"{k}={brief(v)}" for k, v in kwargs.items())
        sig = f"{overload}({arg_str}" + (f", {kw_str}" if kw_str else "") + ")"
        self.calls.append(sig)
        print(f"  [{self.tag}] {overload}")
        print(f"    schema : {schema}")
        print(f"    args   : {arg_str}")
        if kw_str:
            print(f"    kwargs : {kw_str}")
        # 关键：把控制权交回默认路径，实际内核仍会执行（走 CPU/CUDA kernel）
        out = func(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            print(f"    → {brief(out)}")
        print()
        return out


def scene_forward_cpu_no_grad():
    print("[scene] CPU, no_grad, torch.add(a, b)")
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    with torch.no_grad(), TraceDispatchMode("cpu/no_grad"):
        c = torch.add(a, b)
    print(f"  result sum = {c.sum().item():.4f}\n")


def scene_forward_cpu_requires_grad():
    print("[scene] CPU, requires_grad=True, torch.add(a, b, alpha=2)")
    a = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(3, 4, requires_grad=True)
    with TraceDispatchMode("cpu/autograd"):
        c = torch.add(a, b, alpha=2)
    print(f"  grad_fn = {c.grad_fn}")
    print(f"  next_functions = {c.grad_fn.next_functions}\n")

    print("[scene] backward on the above tensor")
    with TraceDispatchMode("cpu/backward"):
        c.sum().backward()
    print(f"  a.grad shape = {tuple(a.grad.shape)}  (should equal a.shape)\n")


def scene_add_scalar():
    """add.Scalar 走 CompositeExplicitAutograd，它内部会调用 add.Tensor。"""
    print("[scene] CPU, torch.add(a, 3.0)  (Scalar 重载)")
    a = torch.randn(3, 4)
    with torch.no_grad(), TraceDispatchMode("cpu/scalar"):
        c = torch.add(a, 3.0)
    print(f"  result sum = {c.sum().item():.4f}\n")


def scene_forward_cuda():
    if not torch.cuda.is_available():
        print("[scene] CUDA skipped (no GPU)\n")
        return
    print("[scene] CUDA, requires_grad=True, torch.add(a, b, alpha=0.5)")
    a = torch.randn(2, 3, device="cuda", requires_grad=True)
    b = torch.randn(2, 3, device="cuda", requires_grad=True)
    with TraceDispatchMode("cuda/autograd"):
        c = torch.add(a, b, alpha=0.5)
    torch.cuda.synchronize()
    print(f"  grad_fn = {c.grad_fn}\n")


if __name__ == "__main__":
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print("=" * 90)
    scene_forward_cpu_no_grad()
    scene_forward_cpu_requires_grad()
    scene_add_scalar()
    scene_forward_cuda()
