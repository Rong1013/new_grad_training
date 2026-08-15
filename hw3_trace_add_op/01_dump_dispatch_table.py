"""
01_dump_dispatch_table.py

静态视角：打印 aten::add.Tensor 在 c10::Dispatcher 里的所有 kernel 注册。

对应 add 算子链路文档的 §7 (Dispatcher 注册)：
  - RegisterSchema.cpp         → schema 定义（TORCH_LIBRARY(aten, m)）
  - RegisterCPU/CUDA/... .cpp  → per-backend kernel（TORCH_LIBRARY_IMPL(aten, KEY, m)）
  - VariableType_*.cpp         → Autograd wrapper
  - TraceType_*.cpp            → Tracer wrapper
  - functorch/vmap/...         → 各种 functional 变换

一句话：这个脚本让 Dispatcher 把它内部的表打出来，眼见为实。
"""

import sys
import textwrap

import torch


OPS = [
    "aten::add.Tensor",   # Tensor + Tensor
    "aten::add.Scalar",   # Tensor + Scalar   (CompositeExplicitAutograd)
    "aten::add.out",      # out= 版本 (structured)
    "aten::add_.Tensor",  # inplace
]


def dump_one(op_name: str) -> None:
    """打印指定 op 的分发表。

    torch._C._dispatch_dump 有两种可能的行为：某些构建里返回字符串，某些构建里
    直接把结果写到 C++ stdout（fd=1）并返回 None。为了两种都能兼容，我们同时：
      1) 取返回值；
      2) fd 级重定向捕获 C++ stdout。
    """
    import os
    import tempfile

    sys.stdout.flush()
    saved_fd = os.dup(1)
    with tempfile.TemporaryFile(mode="w+") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            ret = torch._C._dispatch_dump(op_name)
        finally:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
        tmp.seek(0)
        piped = tmp.read()

    text = ret if isinstance(ret, str) and ret else piped
    print("=" * 90)
    print(f"[dispatch table] {op_name}")
    print("=" * 90)
    print(textwrap.indent(text.rstrip(), "  "))
    print()


def list_registrations_by_key(keys=("CPU", "CUDA", "Autograd", "AutogradCPU",
                                    "AutogradCUDA", "CompositeExplicitAutograd")):
    """对每个 dispatch key，列出该键下所有以 aten::add 开头的注册。

    `_dispatch_print_registrations_for_dispatch_key` 直接把结果写到 C++ stdout
    （fd=1），Python 侧的 sys.stdout 替换 / contextlib.redirect_stdout 都拦不住。
    因此这里用 fd 级重定向：把 fd=1 dup 到一个临时文件，等函数返回后再恢复。
    """
    import os
    import tempfile

    print("=" * 90)
    print("[per-key filter] 只显示 aten::add* 的注册")
    print("=" * 90)
    for k in keys:
        # 保证在读到内容前，先把 python 侧缓冲刷掉，否则会与 C++ 输出交错
        sys.stdout.flush()
        saved_fd = os.dup(1)
        with tempfile.TemporaryFile(mode="w+") as tmp:
            os.dup2(tmp.fileno(), 1)
            try:
                torch._C._dispatch_print_registrations_for_dispatch_key(k)
            except RuntimeError as e:
                os.dup2(saved_fd, 1)
                os.close(saved_fd)
                print(f"  {k}: <skip: {e}>")
                continue
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            tmp.seek(0)
            listing = tmp.read()

        adds = [line for line in listing.splitlines() if "aten::add" in line]
        print(f"\n  --- DispatchKey::{k} ---  (共 {len(listing.splitlines())} 个注册)")
        if not adds:
            print("    (no aten::add* registrations)")
        else:
            for line in adds:
                print("   ", line)


if __name__ == "__main__":
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print()
    for op in OPS:
        dump_one(op)
    list_registrations_by_key()
