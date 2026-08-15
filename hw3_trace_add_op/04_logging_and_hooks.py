"""
04_logging_and_hooks.py

再看两个"低侵入"追踪手法，用来对齐 add 算子链路文档 §8 (Autograd) 与 §7.4 (Dispatcher::call)。

方法 A：TORCH_SHOW_DISPATCH_TRACE=1 环境变量
   PyTorch 内置的 Dispatcher trace，启用后每次 op 分派都会向 stderr 打印
   op 名 + 命中的 dispatch key。这是最贴近 C++ 层的观察方式。

方法 B：nn.Module hook / autograd graph 遍历
   - 用 register_full_backward_hook 抓反向节点
   - 用 grad_fn.next_functions 递归打印 autograd graph，
     确认 AddBackward0 挂在正确的位置。

方法 C：torch.overrides.get_overridable_functions
   查看 torch.add 是否可被 __torch_function__ 拦截（Python 层最外层的 hook）。
"""

import os
import subprocess
import sys
import textwrap

import torch


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- A. TORCH_SHOW_DISPATCH_TRACE ---------- #

def run_with_dispatch_trace() -> None:
    """在子进程里跑一次 torch.add(a, b, alpha=2)，环境变量打开 dispatch trace。

    说明：TORCH_SHOW_DISPATCH_TRACE 只在 debug 构建里生效
        （对应 c10/util/Logging.h 的 SHOW_DISPATCH_TRACE 宏，release wheel 里会被
         预处理掉），因此正式 wheel 上通常什么都不会输出。为了让脚本仍然可用，
    我们同时打印这个"环境要求"的说明。
    """
    print("=" * 90)
    print("[A] TORCH_SHOW_DISPATCH_TRACE=1  —— 观察 Dispatcher 每一次 lookup 命中的 key")
    print("=" * 90)
    script = textwrap.dedent("""
        import torch
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(2, 3, requires_grad=True)
        c = torch.add(a, b, alpha=2)
        c.sum().backward()
    """).strip()
    env = os.environ.copy()
    env["TORCH_SHOW_DISPATCH_TRACE"] = "1"
    res = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=60,
    )
    trace = res.stderr.strip()
    lines = trace.splitlines()
    add_lines = [ln for ln in lines if "add" in ln]
    print(f"  total stderr lines : {len(lines)}")
    print(f"  lines mentioning 'add' : {len(add_lines)}")
    if not add_lines:
        print("  (nothing captured — 说明当前 torch wheel 是 release build，")
        print("   TORCH_SHOW_DISPATCH_TRACE 被预处理掉了。要看到 trace 需要用")
        print("   DEBUG=1 python setup.py develop 自行编译的 debug build。)")
        print("  → 替代方案：用脚本 02_torch_dispatch_trace.py 在 Python 层拦截；")
        print("     或用 03_profiler_trace.py 的 chrome trace 看 kernel 层次。")
    for ln in add_lines[:40]:
        print("   ", ln)
    print()


# ---------- B. Autograd graph ---------- #

def walk_grad_fn(fn, depth=0, seen=None):
    seen = seen or set()
    if fn is None or id(fn) in seen:
        return
    seen.add(id(fn))
    print("  " + "  " * depth + f"- {type(fn).__name__}")
    for nxt, _ in fn.next_functions:
        walk_grad_fn(nxt, depth + 1, seen)


def dump_autograd_graph() -> None:
    print("=" * 90)
    print("[B] Autograd graph —— 从 add 的输出反向遍历 grad_fn.next_functions")
    print("=" * 90)
    a = torch.randn(3, 3, requires_grad=True)
    b = torch.randn(3, 3, requires_grad=True)
    c = torch.add(a, b, alpha=2.5) * 3          # 加个 mul 让图深一点
    d = c.sum()
    print(f"  d.grad_fn = {d.grad_fn}")
    walk_grad_fn(d.grad_fn)

    # 反向 hook：证明 AddBackward0 会真的被 engine 调度
    hooked = []
    handle = c.grad_fn.register_hook(
        lambda gi, go: hooked.append(("AddBackward0", [g.shape for g in go if g is not None]))
    )
    d.backward()
    handle.remove()
    print(f"\n  backward hook fired on AddBackward0: {hooked}")
    print(f"  a.grad shape = {tuple(a.grad.shape)}  b.grad shape = {tuple(b.grad.shape)}\n")


# ---------- C. __torch_function__ ---------- #

class TorchFunctionLogger(torch.overrides.TorchFunctionMode):
    """最外层 Python 拦截。torch.add 会在这里先命中，再进入 C++ Dispatcher。"""
    def __init__(self):
        super().__init__()
        self.log = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # func 可能是 python function、method_descriptor、slot_wrapper 等；
        # 取一个稳妥的显示名
        name = getattr(func, "__qualname__", None) or getattr(func, "__name__", str(func))
        module = getattr(func, "__module__", "") or ""
        self.log.append(f"{module}.{name}" if module else name)
        return func(*args, **kwargs)


def show_torch_function() -> None:
    print("=" * 90)
    print("[C] TorchFunctionMode —— Python 层最外层的拦截点")
    print("=" * 90)
    a = torch.randn(4)
    b = torch.randn(4)
    with TorchFunctionLogger() as logger:
        c = torch.add(a, b)
        d = a + b       # 走 __add__，最终也会命中 aten::add.Tensor
        e = a.add(b)
    print(f"  logged calls: {logger.log}")
    print(f"  results equal: {torch.equal(c, d) and torch.equal(c, e)}\n")


if __name__ == "__main__":
    print("torch:", torch.__version__)
    print()
    run_with_dispatch_trace()
    dump_autograd_graph()
    show_torch_function()
