# HW3 — 追踪一次 `torch.add()` 的执行路径

## 0. 目录结构

```
hw3_trace_add_op/
├── 01_dump_dispatch_table.py    # 静态视角：dump c10::Dispatcher 里 aten::add.* 的注册表
├── 02_torch_dispatch_trace.py   # 用 TorchDispatchMode 拦截前向 / 反向 op 序列
├── 03_profiler_trace.py         # torch.profiler 采样，导出 chrome trace + FLOPs
├── 04_logging_and_hooks.py      # TorchFunctionMode + autograd graph 遍历 + 反向 hook
├── run_all.sh                   # 一键跑完并把输出写到 logs/
├── logs/                        # run_all.sh 产出的运行日志
├── add_forward_trace_cpu.json   # profiler 生成的 chrome trace，可导入 chrome://tracing
├── add_forward_trace_cuda.json
├── add_backward_trace_cpu.json
├── add_backward_trace_cuda.json
└── README.md                    # 本文
```

## 1. 快速开始

```bash
cd hw3_trace_add_op
bash run_all.sh
```

单独跑：
```bash
python3 01_dump_dispatch_table.py   | less
python3 02_torch_dispatch_trace.py
python3 03_profiler_trace.py
python3 04_logging_and_hooks.py
```

Chrome trace 查看：Chrome 打开 `chrome://tracing`，加载 `add_forward_trace_cuda.json` 即可看到时间轴。

---

## 2. 一次 `torch.add(a, b, alpha=α)` 的实际调用链

配合 4 个脚本的实际输出，`add` 算子的完整执行序列如下：

```
Python: torch.add(a, b, alpha=α)
   │
   ├── [04-C] TorchFunctionMode 拦截           ← 最外层 Python hook
   │      logged: torch._VariableFunctionsClass.add
   │
   ▼
python_torch_functions.cpp: THPVariable_add    (@generated)
   │   - PythonArgParser 解析 args/kwargs
   │   - AutoNoGIL 释放 GIL
   ▼
ATen/ops/add.h: at::add(self, other, alpha)    (@generated)
   ▼
Operators_*.cpp: at::_ops::add_Tensor::call
   │   - findSchemaOrThrow -> TypedOperatorHandle
   ▼
c10::Dispatcher::call
   │
   ├── [01] dispatch 表命中顺序（由 DispatchKeySet 决定）：
   │       Autograd -> CPU/CUDA -> (Sparse/Nested/MPS/... 未命中)
   │   看 01 输出可以清楚看到这些 key 在 aten::add.Tensor 上都注册了 kernel
   │
   ▼
VariableType_2.cpp: VariableType::add_Tensor   ← Autograd wrapper
   │   - compute_requires_grad(a, b) == True → 构造 AddBackward0
   │   - grad_fn->alpha = α
   │   - collect_next_edges(a, b) -> (AccumulateGrad@a, AccumulateGrad@b)
   │   - AutoDispatchBelowADInplaceOrView guard
   │   - redispatch(ks & after_autograd_keyset, ...)  ← 剥掉 Autograd 键
   │
   │   [02] TorchDispatchMode 就是挂在 "Autograd 之后" 这一层看到 aten.add.Tensor
   │
   ▼
RegisterCPU.cpp / RegisterCUDA.cpp: wrapper_<KEY>_add_Tensor  (@generated)
   │   - structured_ufunc_add_<KEY>_functional op;
   │   - op.meta(self, other, alpha)  → build_borrowing_binary_op (BinaryOps.cpp:151)
   │       * 广播 + type promotion + 分配 out
   │   - op.impl(self, other, alpha, out)
   │
   ▼
UfuncCPU_add.cpp / UfuncCUDA_add.cu: TORCH_IMPL_FUNC(ufunc_add_<KEY>)  (@generated)
   │   - add_stub(device_type(), *this, alpha)  ← DispatchStub
   │
   ▼
UfuncCPUKernel_add.cpp: add_kernel(iter, alpha)   ← REGISTER_DISPATCH(add_stub, ...)
   │   - AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, ..., [&]() {
   │       cpu_kernel_vec(iter, scalar_op, vec_op);
   │     });
   │   [03] profiler 在这里看到 aten::add 事件占了主要 CPU/CUDA 时间
   │
   ▼
ufunc::add(a, b, α) = a + α * b     ← aten/src/ATen/native/ufunc/add.h（唯一手写数学）
   │
   ▼
返回 Tensor
   │
   ▲ set_history(result, AddBackward0)   ← 回到 VariableType 层
   │
   ▲ wrap 成 THPVariable，返回 Python
```

反向 (`c.sum().backward()`)：

```
autograd Engine
   ▼ SumBackward0            (由 aten::sum → SumBackward0 挂钩)
   ▼ AddBackward0::apply     ← [04-B] 反向 hook 抓到；[02] TorchDispatchMode 会看到底下的 aten.mul.Scalar / aten.expand
       grad_a = handle_r_to_c(dtype_a, grad)
       grad_b = handle_r_to_c(dtype_b, maybe_multiply(grad, alpha.conj()))
       ↑ 这两行由 tools/autograd/derivatives.yaml:229 生成
   ▼ AccumulateGrad → 累加进 a.grad / b.grad
```

---

## 3. 四个脚本对应链路文档里的哪一段

| 脚本 | 观察点 | 对应 `pytorch-add-operator-registration.md` 章节 |
| :--- | :--- | :--- |
| `01_dump_dispatch_table.py` | Dispatcher 的静态注册表（每个 dispatch key 的 kernel 位置） | §7 Dispatcher 注册（RegisterSchema.cpp + RegisterCPU/CUDA/... + TORCH_LIBRARY_IMPL） |
| `02_torch_dispatch_trace.py` | 运行时 Autograd 之后每一次 op 分派（含反向图内部序列） | §7.4 Dispatcher::call + §10 端到端调用链 |
| `03_profiler_trace.py` | kernel 级别的耗时/输入 shape，含 CPU vs CUDA、前向 vs 反向 | §5 CPU/CUDA kernel + §10 |
| `04_logging_and_hooks.py` | 最外层 Python hook + Autograd 图结构 + 反向 hook | §9 Python 绑定 + §8 Autograd |

---

## 4. 实际输出摘要（跑一次的关键片段）

### 4.1 `01`：aten::add.Tensor 的注册全表

```
[dispatch table] aten::add.Tensor
  name: aten::add.Tensor
  schema: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  debug: registered at .../RegisterSchema.cpp:6
  alias analysis kind: FROM_SCHEMA
  MkldnnCPU:        RegisterMkldnnCPU_0.cpp:162       ← 手写
  ZeroTensor:       RegisterZeroTensor_0.cpp:114      ← 手写
  Tracer:           TraceType_2.cpp:18340             ← @generated
  FuncTorchBatched: BatchRulesBinaryOps.cpp:352
  Batched:          LegacyBatchingRegistrations.cpp:1079
  CPU:              RegisterCPU_0.cpp:1309            ← wrapper_CPU_add_Tensor (§7.2)
  CUDA:             RegisterCUDA_0.cpp:2494           ← wrapper_CUDA_add_Tensor
  Meta:             torch/_meta_registrations.py:50   ← Python 优先
  Meta (inactive):  RegisterMeta_0.cpp:1158
  SparseCPU/CUDA/Meta/Csr* : Register{Sparse,SparseCsr}*.cpp   ← 手写 (§5.4)
  NestedTensorCPU/CUDA/HPU : RegisterNestedTensor*.cpp         ← 手写
  Autograd[alias]:  VariableType_2.cpp:20465          ← @generated (§8.2)
  CompositeExplicitAutogradNonFunctional[alias]: RegisterCompositeExplicitAutogradNonFunctional_0.cpp:1374
```

这一段完美对应文档 §7.2「Per-backend kernel 注册」：**每一行 `KEY: registered at ...` 就是一个 `TORCH_LIBRARY_IMPL(aten, KEY, m) { m.impl("add.Tensor", TORCH_FN(...)); }`。**

对比 `aten::add.Scalar`：

```
[dispatch table] aten::add.Scalar
  CompositeExplicitAutograd[alias]: RegisterCompositeExplicitAutograd_0.cpp:2030
```

没有 CPU/CUDA 单独注册 —— 因为 yaml 里写的是 `dispatch: CompositeExplicitAutograd: add`，一个 kernel 通吃所有 backend（§1）。

### 4.2 `02`：TorchDispatchMode 看到的 op 序列

**CPU + requires_grad + alpha=2 前向：**
```
[cpu/autograd] aten.add.Tensor
  args   : Tensor(shape=(3,4), req_grad=True), Tensor(shape=(3,4), req_grad=True)
  kwargs : alpha=2
grad_fn = <AddBackward0>
next_functions = (AccumulateGrad, AccumulateGrad)
```

**反向 `c.sum().backward()` 底层拆成 8 个原子 op：**
```
aten.sum.default              → SumBackward0
aten.ones_like.default        → 分配上游 grad
aten.expand.default           → 广播成 (3,4)
aten.mul.Scalar               ← AddBackward0 里 maybe_multiply(grad, alpha)，alpha=2
aten.new_empty_strided        → AccumulateGrad 分配
aten.copy_.default            → 写入 a.grad / b.grad
aten.detach.default × 2
```

`aten.mul.Scalar` 就是 `derivatives.yaml:229` 里 `maybe_multiply(grad, alpha.conj())` 展开的结果（文档 §8.1）。

**CUDA 场景：**
```
[cuda/autograd] aten.add.Tensor
  args : Tensor(dev=cuda:0, ...), Tensor(dev=cuda:0, ...)  kwargs: alpha=0.5
```
Dispatcher 直接命中 `AutogradCUDA → CUDA` 路径。

### 4.3 `03`：Profiler 看到的时间/内核

**CPU 前向（5 次 1024×1024 add）：**
```
aten::add        28.60%   11.832ms   [[1024,1024],[1024,1024],[]]   5.243 MFLOPs
py::torch.add    (record_function 加的用户标签)
```

**CUDA 前向：**
```
aten::add                                                 95.834us
void at::native::vectorized_elementwise_kernel<...>       54.875us   ← 真正的 GPU kernel
```
GPU 上真正干活的是 `vectorized_elementwise_kernel`，它就是 §5.3 里 `gpu_kernel(iter, CUDAFunctor_add<scalar_t>{})` 通过 `CUDAFunctorOnOther_add` 等 functor 启动出来的。

**CUDA 反向：**
```
py::c.sum().backward()          10.885ms   ← 总墙钟
aten::sum                       66.074us   ← SumBackward0 的正向对应
aten::add_                      51.196us   ← AccumulateGrad 里的 in-place add
void at::native::reduce_kernel<...>         48.317us
```

打开 `add_forward_trace_cuda.json`（chrome://tracing）可以看到 CPU 侧的 `aten::add` 时间条与 GPU 侧的 `vectorized_elementwise_kernel` 时间条之间有明显时间差 —— 那就是 launch overhead。

### 4.4 `04`：Autograd 图与 Python 层拦截

**TorchFunctionMode（Python 最外层）：**
```
logged calls: [
  'torch._VariableFunctionsClass.add',   ← torch.add(a, b)
  'TensorBase.add',                       ← a + b   （Python __add__ 走 Tensor.add）
  'TensorBase.add',                       ← a.add(b)
]
```
三种写法在 Python 侧就分道了，但最终都会汇合到 C++ 里同一个 `aten::add.Tensor` schema。

**Autograd 图（`(a+b)*3` 后 sum）：**
```
- SumBackward0
  - MulBackward0
    - AddBackward0                ← α=2.5 保存在这个节点里
      - AccumulateGrad × 2
```
**反向 hook 触发：**
```
backward hook fired on AddBackward0: [('AddBackward0', [torch.Size([3, 3])])]
```
证明 `AddBackward0::apply({grad})` 确实被 engine 调度，符合文档 §8.2 里生成的 `AddBackward0::apply` 逻辑。

---

## 5. 常见问题

**Q1：为什么 `TORCH_SHOW_DISPATCH_TRACE=1` 什么都不打印？**
`SHOW_DISPATCH_TRACE` 宏只在 debug 构建里生效（`c10/util/Logging.h`）。官方 release wheel 里被预处理掉了。要看到 C++ 层的分派 trace 必须 `DEBUG=1 python setup.py develop` 自建一个 debug build。脚本 04-A 会打印说明。

**Q2：为什么 `02_torch_dispatch_trace.py` 看不到 `AutogradXPU -> CPU` 两层？只看到一次 `aten.add.Tensor`。**
`TorchDispatchMode` 挂在 `Python` 键的 fallback 里，位于 `Autograd` 之下、backend 之上。它看到的**已经是 Autograd 之后 redispatch 下来的那次调用**。要看到 Autograd 那一层，得用 `04-B` 的 `grad_fn.next_functions` + `register_hook`，或者用 profiler 的时间线（每个 kernel 名字前面会有 "autograd::engine::evaluate_function: AddBackward0" 这一层）。

**Q3：`aten::add.Scalar` 为什么在 CUDA/CPU 键上没显示？**
它注册在 `CompositeExplicitAutograd[alias]` 上，Dispatcher 对所有非 Autograd 的 backend 用这一份实现（内部 wrap 成 Tensor 后再调 `aten::add.Tensor`）。见文档 §1 和 §5.4。

**Q4：chrome trace 里的 `hipMalloc`/`hipLaunchKernel`/`Memset` 是什么？**
分别是 HIP runtime 的显存分配、kernel 启动、异步内存写零（`AccumulateGrad` 初始化 grad 用）。CUDA 版本对应 `cudaMalloc` / `cudaLaunchKernel` / `cudaMemsetAsync`，只是当前 GPU 是 ROCm 后端。

---

## 6. 下一步（可扩展方向）

- 把 `03` 里的 chrome trace 用 [Perfetto UI](https://ui.perfetto.dev) 打开，可以做时间轴对齐分析
- 用 `TorchDispatchMode` 做**自定义 kernel 拦截**：在 `__torch_dispatch__` 里改写 `aten.add.Tensor` 为 `aten.sub.Tensor + 2*b`，验证是否等价（Python 层就能干预 dispatcher）
- 用 `torch.library` 注册一个 backend override：
  ```python
  from torch.library import Library
  lib = Library("aten", "IMPL", "CPU")
  lib.impl("add.Tensor", my_add, "CPU")
  ```
  这对应文档 §11「加新 backend」的 Python 版本

## 7. 参考

- 文档：[`pytorch-add-operator-registration.md`](pytorch-add-operator-registration.md)
- 源码：[aten/src/ATen/native/BinaryOps.cpp](pytorch/aten/src/ATen/native/BinaryOps.cpp), [aten/src/ATen/native/ufunc/add.h](pytorch/aten/src/ATen/native/ufunc/add.h)
- 分发器：[aten/src/ATen/core/dispatch/Dispatcher.h](pytorch/aten/src/ATen/core/dispatch/Dispatcher.h), [torch/library.h](pytorch/torch/library.h)
- 自动求导规则：[tools/autograd/derivatives.yaml:229](pytorch/tools/autograd/derivatives.yaml#L229)
