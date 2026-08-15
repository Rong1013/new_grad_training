# PyTorch Operator & Kernel 注册链路 —— 以 `add` 算子为例

> 本文以 `torch.add` 为主线，梳理 PyTorch 中一个算子从**声明**、**meta / 形状推断**、**dispatch stub**、**CPU/CUDA kernel 实现**、**代码生成**、**Dispatcher 注册**、**Autograd 包装**到**Python 绑定**的完整链路。
>
> 涉及的仓库路径均相对于 `pytorch/` 根目录；示例中的生成文件片段取自安装好的 wheel（`torch/include/ATen/ops/...`），对应源码中的模板（`aten/src/ATen/templates/*` + `torchgen/dest/*.py`）。

---

## 0. 全景图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  aten/src/ATen/native/native_functions.yaml           <-- 唯一"schema 事实源"  │
│    - func: add.Tensor(...)   structured_delegate: add.out                    │
│    - func: add.out(...)      structured: True   ufunc_inner_loop: ...        │
└──────────────────────────────────────────────────────────────────────────────┘
                │                                          │
                │ torchgen/gen.py 读取                     │ tools/autograd/gen_*.py 读取
                │ (torchgen/dest/*.py 中定义生成器)         │ + tools/autograd/derivatives.yaml
                ▼                                          ▼
┌──────────────────────────────┐          ┌────────────────────────────────────┐
│ 生成 ATen 前端 & 分发          │          │ 生成 Autograd 包装 & Python 绑定    │
│  - ATen/ops/add.h            │          │  - torch/csrc/autograd/generated/  │
│  - ATen/ops/add_ops.h        │          │       VariableType_*.cpp           │
│  - ATen/ops/add_native.h     │          │       Functions.{h,cpp}            │
│  - ATen/ops/add_meta.h       │          │       python_torch_functions.cpp   │
│  - ATen/ops/add_*_dispatch.h │          │       python_variable_methods.cpp  │
│  - RegisterCPU.cpp,          │          │                                    │
│    RegisterCUDA.cpp, ...     │          │                                    │
│    RegisterSchema.cpp        │          │                                    │
│    RegisterBackendSelect.cpp │          │                                    │
│    Operators_*.cpp           │          │                                    │
│    UfuncCPU_add.cpp          │          │                                    │
│    UfuncCPUKernel_add.cpp    │          │                                    │
│    UfuncCUDA_add.cu          │          │                                    │
└──────────────────────────────┘          └────────────────────────────────────┘
                │                                          │
                └───────────────┬──────────────────────────┘
                                ▼
                ┌────────────────────────────────────┐
                │   c10::Dispatcher (运行时)          │
                │   op.lookup(DispatchKeySet) → 内核 │
                └────────────────────────────────────┘
```

一切从 `native_functions.yaml` 出发，`torchgen/`（ATen）与 `tools/autograd/`（Autograd + Python）在**编译期**读取这份 yaml + 若干模板生成 C++/CUDA 源码；运行时通过 `c10::Dispatcher` 根据 `DispatchKeySet` 选择实际内核。

---

## 1. Schema 事实源：`native_functions.yaml`

文件：`aten/src/ATen/native/native_functions.yaml` (第 552–632 行)

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS, SparseMeta: add_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: add_sparse_csr
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: [core, pointwise]

- func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
  device_check: NoCheck
  variants: method
  structured_delegate: add.out
  dispatch:
    SparseCPU, SparseCUDA, ...: add_sparse_
    ...

- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck
  structured: True                          # <── 是"结构化算子"，走 meta+impl 两段式
  structured_inherits: TensorIteratorBase
  ufunc_inner_loop:                         # <── 使用 ufunc 代码生成
    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)
    ScalarOnly: add (Bool)
  dispatch:
    SparseCPU, SparseMeta: add_out_sparse_cpu
    SparseCUDA: add_out_sparse_cuda
    SparseMPS: add_out_sparse_mps
    ...
    MPS: add_out_mps
    MTIA: add_out_mtia
  tags: pointwise

- func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  dispatch:
    CompositeExplicitAutograd: add          # <── 与 Tensor 版本不同，走 Composite kernel
```

关键字段：

| 字段 | 含义 |
| :--- | :--- |
| `func` | 算子 schema，`name.overload(args) -> ret`，是唯一 ID |
| `structured: True` | 这是"结构化算子"，与 `TORCH_META_FUNC` / `TORCH_IMPL_FUNC` 配对 |
| `structured_delegate: add.out` | 该重载不写 impl，直接复用 `add.out` 里的 meta+impl（`add.Tensor` / `add_` 都委托到 `add.out`） |
| `ufunc_inner_loop` | 只需给 CPU/CUDA 的**逐元素标量函数**，dispatch stub、kernel 由 codegen 自动生成 |
| `dispatch:` | 指定某些 dispatch key（Sparse/MPS/MkldnnCPU/…）需要**手写** kernel 的符号名 |
| `variants` | 是否生成 `at::add(...)` 函数和 `Tensor::add(...)` 成员方法 |
| `tags: pointwise` | 元数据，供 functorch / torch.compile 等使用 |

`add` 的四组重载：`.Tensor` / `.Tensor_out(add.out)` / `.Scalar` / `.Scalar_out`（`autogen: add.Scalar_out`）；`in-place` 版本 `add_.Tensor` 通过 `structured_delegate` 复用 `add.out`，无需重写。

---

## 2. Meta 函数：形状 / dtype 推断

文件：`aten/src/ATen/native/BinaryOps.cpp` (第 149–156 行)

```cpp
namespace at::meta {

TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

}  // namespace at::meta
```

`TORCH_META_FUNC2(add, Tensor)` 展开后是 `structured_add_Tensor::meta`，其类声明由 codegen 生成到 `ATen/ops/add_meta.h`：

```cpp
// torch/include/ATen/ops/add_meta.h  (@generated)
namespace at::meta {
struct TORCH_API structured_add_Tensor : public TensorIteratorBase {
  void meta(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
};
}
```

`build_borrowing_binary_op` 会调用 `TensorIterator::build`，完成：
- 广播（broadcast）
- 结果 dtype 提升（type promotion）
- 结果 shape / stride 计算，并调用 `set_output_raw_strided` 分配 out tensor（若 `out=` 已给则复用）

`alpha_check` 保证 `alpha` 的类型与结果 dtype 兼容（`native/BinaryOps.h:16`）。

Meta 函数**不做任何计算**，只负责"形状+dtype 校验+输出分配"，让 CPU/CUDA/MPS 等 backend 复用同一份形状推断逻辑。

---

## 3. Dispatch Stub：`add_stub`

### 3.1 声明（同一个符号，所有 backend 共享）

文件：`aten/src/ATen/native/BinaryOps.h` (第 45–56 行)

```cpp
using structured_binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);

// NB: codegenned  ← 由 ufunc codegen 定义
DECLARE_DISPATCH(structured_binary_fn_alpha, add_stub)
```

`DECLARE_DISPATCH` 展开成一个可按设备切换实现的全局函数对象；`add_stub(device_type, iter, alpha)` 会根据 `device_type` 派发到 CPU/CUDA 各自注册的 kernel。

### 3.2 定义

**注意**：`add_stub` 的 `DEFINE_DISPATCH` 不在 `BinaryOps.cpp` 里（那里手写的是 `add_clamp_stub`、`mul_stub` 等），而是**由 codegen 生成到 `UfuncCPU_add.cpp`** 里（因为 `add.out` 用了 `ufunc_inner_loop`）。见第 5 节。

### 3.3 与 `TORCH_IMPL_FUNC` 的关系

对**非 ufunc 的结构化算子**，作者会手写：

```cpp
TORCH_IMPL_FUNC(sub_out) (Tensor& self, Tensor& other, Scalar alpha, Tensor& result) {
  add_stub(device_type(), *this, -alpha);
}
```
（`BinaryOps.cpp:433`）

而 `add.out` 因为在 yaml 里声明了 `ufunc_inner_loop`，`TORCH_IMPL_FUNC(add_out)` 的**实现体本身也是 codegen 生成**的（在 `UfuncCPU_add.cpp` / `UfuncCUDA_add.cu` 里）。

---

## 4. Ufunc：真正的标量数学

文件：`aten/src/ATen/native/ufunc/add.h`（**唯一手写的算法源文件**）

```cpp
namespace at::native::ufunc {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T
add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
  return self + alpha * other;
}

#if !defined(__CUDACC__) && !defined(__HIPCC__)
using vec::Vectorized;
template <typename T>
C10_ALWAYS_INLINE Vectorized<T>
add(Vectorized<T> self, Vectorized<T> other, Vectorized<T> alpha) {
  return vec::fmadd(other, alpha, self);   // 一条 FMA
}
#endif

} // namespace at::native::ufunc
```

只有 10 行左右——CPU 标量、CPU 向量（AVX/AVX-512/NEON）、CUDA/HIP 都调用同一个 `ufunc::add`。CPU 向量化走 `ATen/cpu/vec` 里的 `Vectorized<T>`，CUDA 通过后面提到的 codegen 生成 functor。

---

## 5. Codegen：Ufunc 展开到 CPU / CUDA

代码生成器：`torchgen/api/ufunc.py`、`torchgen/dest/ufunc.py`
模板：`aten/src/ATen/templates/UfuncCPU.cpp`、`UfuncCPUKernel.cpp`、`UfuncCUDA.cu`

### 5.1 CPU 骨架（`UfuncCPU.cpp` 模板）

```cpp
// UfuncCPU_add.cpp (@generated)
#include <ATen/native/DispatchStub.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>

namespace at {
namespace meta {
  // 已在第 2 节生成过 structured_add_Tensor
}
namespace native {

// DEFINE_DISPATCH(add_stub)      <── 由 codegen 展开
// TORCH_IMPL_FUNC(ufunc_add_CPU)(...)  <── 调用 add_stub(device_type(), *this, alpha)

}}
```

即 `TORCH_IMPL_FUNC(add_out)` 展开成 `structured_ufunc_add_CPU::impl`（可从生成头 `add_native.h` 反查其存在）：

```cpp
// torch/include/ATen/ops/add_native.h (@generated)
namespace at::native {
struct TORCH_API structured_ufunc_add_CPU : public at::meta::structured_add_Tensor {
  void impl(const at::Tensor& self, const at::Tensor& other,
            const at::Scalar& alpha, const at::Tensor& out);
};
struct TORCH_API structured_ufunc_add_CUDA : public at::meta::structured_add_Tensor {
  void impl(...);
};
// 手写的其它 dispatch key 的原型：
TORCH_API at::Tensor NestedTensor_add_Tensor(...);
TORCH_API at::Tensor add_sparse(...);
TORCH_API at::Tensor& add_out_sparse_cpu(...);
...
}
```

### 5.2 CPU 内层循环（`UfuncCPUKernel.cpp` 模板）

生成的 `UfuncCPUKernel_add.cpp` 大致长这样：

```cpp
#include <ATen/native/ufunc/add.h>
#include <ATen/native/cpu/Loops.h>

namespace at::native {
namespace {
// 每种 dtype 一份，用 AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16, ...) 分派
void add_kernel(TensorIteratorBase& iter, const Scalar& alpha) {
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "add_cpu", [&]() {
    auto alpha_v = alpha.to<scalar_t>();
    cpu_kernel_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t {
        return ufunc::add(a, b, alpha_v);
      },
      [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
        return ufunc::add(a, b, Vectorized<scalar_t>(alpha_v));
      });
  });
}
}
REGISTER_DISPATCH(add_stub, &add_kernel);   // <── 将 add_kernel 注册到 add_stub@CPU
}
```

即 **`add_stub` 的 CPU 实现由 `REGISTER_DISPATCH` 注册**，`REGISTER_DISPATCH` 展开为把函数指针写入 `add_stub` 的 CPU 槽位；如果开启了 AVX2/AVX-512 分片，构建系统会为每个 ISA 单独编译一份，然后运行时按 CPU 能力选择。

### 5.3 CUDA（`UfuncCUDA.cu` 模板 + `torchgen/dest/ufunc.py`）

CUDA 版把 ufunc 打包成 functor（`torchgen/dest/ufunc.py:52-64` 注释里画得很清楚）：

```cpp
// UfuncCUDA_add.cu (@generated)
template <typename scalar_t>
struct CUDAFunctorOnSelf_add {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t other_, alpha_;
  CUDAFunctorOnSelf_add(opmath_t other, opmath_t alpha)
    : other_(other), alpha_(alpha) {}
  __device__ scalar_t operator()(scalar_t self) const {
    return ufunc::add(static_cast<opmath_t>(self), other_, alpha_);
  }
};
// 类似还有 CUDAFunctorOnOther_add / CUDAFunctor_add

void add_kernel_cuda(TensorIteratorBase& iter, const Scalar& alpha) {
  AT_DISPATCH_..._TYPES(..., "add_cuda", [&]() {
    // 走 gpu_kernel(iter, functor) / gpu_kernel_with_scalars
  });
}
REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
```

`opmath_t` 用于把 `half` / `bfloat16` 提升到 `float` 做累加，避免精度损失。

### 5.4 手写 backend kernel

`native_functions.yaml` 里 `dispatch:` 直接写了函数名的 backend（Sparse*, Mkldnn*, MPS, MTIA, ZeroTensor, NestedTensor*）都是**手写**：

- `SparseCPU/SparseCUDA`：`aten/src/ATen/native/sparse/SparseTensorMath.cpp` 里 `add_out_sparse_cpu`
- `MPS`：`aten/src/ATen/native/mps/operations/BinaryOps.mm:262` `TORCH_IMPL_FUNC(add_out_mps)`
- `NestedTensor*`：`aten/src/ATen/native/nested/NestedTensorBinaryOps.cpp`
- `MkldnnCPU`：`aten/src/ATen/native/mkldnn/BinaryOps.cpp`
- `add.Scalar` 用 `CompositeExplicitAutograd: add` —— 内部转成 `at::add(self, wrapped_scalar_tensor(other), alpha)`（`BinaryOps.cpp` 里的 `add(Tensor, Scalar, Scalar)`）

---

## 6. 生成的 ATen 前端头（每算子一份，`AT_PER_OPERATOR_HEADERS`）

### 6.1 `ATen/ops/add.h` —— 用户可见 API

```cpp
// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
inline at::Tensor add(const at::Tensor& self, const at::Tensor& other,
                      const at::Scalar& alpha=1) {
  return at::_ops::add_Tensor::call(self, other, alpha);
}
// aten::add.out(...)
inline at::Tensor& add_out(at::Tensor& out, const at::Tensor& self,
                           const at::Tensor& other, const at::Scalar& alpha=1) {
  return at::_ops::add_out::call(self, other, alpha, out);
}
inline at::Tensor add(const at::Tensor& self, const at::Scalar& other,
                      const at::Scalar& alpha=1) {
  return at::_ops::add_Scalar::call(self, other, alpha);
}
```

`variants: function` 生成 `at::add`；`variants: method` 生成 `Tensor::add`（见下）。

### 6.2 `ATen/ops/add_ops.h` —— Dispatcher 入口

```cpp
namespace at::_ops {
struct TORCH_API add_Tensor {
  using schema = at::Tensor(const at::Tensor&, const at::Tensor&, const at::Scalar&);
  static constexpr const char* name = "aten::add";
  static constexpr const char* overload_name = "Tensor";
  static constexpr const char* schema_str =
      "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor";
  static at::Tensor call(const at::Tensor&, const at::Tensor&, const at::Scalar&);
  static at::Tensor redispatch(c10::DispatchKeySet, const at::Tensor&, ...);
};
struct TORCH_API add_out { ... };
struct TORCH_API add_Scalar { ... };
struct TORCH_API add__Tensor { ... };    // inplace
struct TORCH_API add_Scalar_out { ... };
struct TORCH_API add__Scalar { ... };
}
```

### 6.3 `ATen/ops/add_meta.h` / `add_native.h` / `add_cpu_dispatch.h`

见第 2、5 节。`add_cpu_dispatch.h` 提供绕过 Autograd 的直接 backend 调用：

```cpp
namespace at::cpu {
TORCH_API at::Tensor  add(const at::Tensor& self, const at::Tensor& other,
                          const at::Scalar& alpha=1);
TORCH_API at::Tensor& add_out(at::Tensor& out, ...);
TORCH_API at::Tensor& add_(at::Tensor& self, ...);
}
```

### 6.4 `Operators_*.cpp` —— `_ops::add_Tensor::call` 的实现

`torchgen/gen.py:651-676`：

```cpp
// Operators_2.cpp (@generated, sharded)
static C10_NOINLINE c10::TypedOperatorHandle<add_Tensor::schema>
create_add_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_Tensor::name, add_Tensor::overload_name)
      .typed<add_Tensor::schema>();
}

at::Tensor add_Tensor::call(const at::Tensor& self, const at::Tensor& other,
                            const at::Scalar& alpha) {
  static auto op = create_add_Tensor_typed_handle();
  return op.call(self, other, alpha);      // <── 走进 Dispatcher
}

at::Tensor add_Tensor::redispatch(c10::DispatchKeySet ks, ...) {
  static auto op = create_add_Tensor_typed_handle();
  return op.redispatch(ks, self, other, alpha);
}
```

`typed<schema>()` 把 `OperatorHandle` 转成强类型的 `TypedOperatorHandle`，避免每次调用都做 `IValue` 装箱。

---

## 7. Dispatcher 注册

### 7.1 Schema 注册（`aten` 命名空间的定义）

模板：`aten/src/ATen/templates/RegisterSchema.cpp`

```cpp
TORCH_LIBRARY(aten, m) {
  m.def("add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
  m.def("add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)");
  m.def("add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor");
  ...
}
```

`TORCH_LIBRARY` 只出现**一次**（`torch/library.h:982`）——负责创建 `OperatorHandle`，把 schema 存进全局 `c10::Dispatcher::realSingleton()`。

### 7.2 Per-backend 的 kernel 注册

模板：`aten/src/ATen/templates/RegisterDispatchDefinitions.ini` + `torchgen/gen.py:1635`：

```cpp
// RegisterCPU.cpp (@generated, sharded)
namespace at {
namespace {
// -- 结构化算子的 wrapper (StructuredRegisterDispatchKey 生成) --
struct structured_ufunc_add_CPU_functional final : public at::native::structured_ufunc_add_CPU {
  ...  // 继承 meta+impl，处理 out=None 时的输出分配
};

at::Tensor wrapper_CPU_add_Tensor(const at::Tensor& self, const at::Tensor& other,
                                  const at::Scalar& alpha) {
  structured_ufunc_add_CPU_functional op;
  op.meta(self, other, alpha);      // 走第 2 节的 meta
  op.impl(self, other, alpha, op.outputs_[0]);  // 走第 5 节的 impl (→ add_stub → REGISTER_DISPATCH)
  return std::move(op.outputs_[0]);
}
at::Tensor& wrapper_CPU_add_out_out(...)  { ... }
at::Tensor& wrapper_CPU_add__Tensor(...)  { ... }
}  // anonymous

TORCH_LIBRARY_IMPL(aten, CPU, m) {                     // <── 注册 CPU kernel
  m.impl("add.Tensor", TORCH_FN(wrapper_CPU_add_Tensor));
  m.impl("add.out",    TORCH_FN(wrapper_CPU_add_out_out));
  m.impl("add_.Tensor",TORCH_FN(wrapper_CPU_add__Tensor));
}

namespace cpu {
at::Tensor add(...) { return wrapper_CPU_add_Tensor(...); }   // add_cpu_dispatch.h 的实现
at::Tensor& add_out(...) { ... }
}
}
```

- `RegisterCUDA.cpp`、`RegisterMPS.cpp`、`RegisterSparseCPU.cpp`、`RegisterNestedTensorCPU.cpp` … 结构相同。
- `RegisterCompositeExplicitAutograd.cpp` 里注册 `add.Scalar`（因为 yaml 里 `add.Scalar` 声明 `CompositeExplicitAutograd: add`）。
- `RegisterBackendSelect.cpp`（模板 `aten/src/ATen/templates/RegisterBackendSelect.cpp`）注册 `BackendSelect` 键，处理只有 device/dtype 参数、没有 Tensor 时的分派入口（`add` 不需要，但工厂函数需要）。

### 7.3 `TORCH_LIBRARY_IMPL` 宏

`torch/library.h:1072`：

```cpp
#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)                             \
  static void TORCH_LIBRARY_IMPL_init_##ns##_##k##_##uid(torch::Library&); \
  static const torch::detail::TorchLibraryInit                         \
    TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_##uid(                 \
      torch::Library::IMPL,                                            \
      &TORCH_LIBRARY_IMPL_init_##ns##_##k##_##uid,                     \
      #ns,                                                             \
      std::make_optional(c10::DispatchKey::k),                         \
      __FILE__, __LINE__);                                             \
  void TORCH_LIBRARY_IMPL_init_##ns##_##k##_##uid(torch::Library& m)
```

- 每个 `TORCH_LIBRARY_IMPL(aten, CPU, m) { ... }` 都会生成一个**静态对象**，构造时把 `m.impl(...)` 中的 kernel 写入 Dispatcher 的 per-op 表；因此 kernel 注册是**动态库加载阶段**完成的（DSO 静态初始化）。
- `m.impl("add.Tensor", TORCH_FN(fn))` 最终调用到 `c10::impl::OperatorEntry::registerKernel(dispatch_key, KernelFunction::makeFromUnboxedRuntimeFunction(&fn))`。

### 7.4 运行时分派：`c10::Dispatcher::call`

`aten/src/ATen/core/dispatch/Dispatcher.h:776`：

```cpp
template <class Return, class... Args>
Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  // 1. 从参数中抽出 DispatchKeySet：遍历 args 里的 Tensor，取 device/layout 对应的 key，
  //    与 TLS 的 include/exclude 集合合并
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
      .getDispatchKeySetUnboxed<Args...>(args...);

  // 2. 查表：从 per-op 的 dispatch table 里，按 DispatchKeySet 里最高优先级的键取 kernel
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);

  // 3. 调 kernel（strongly-typed，无装箱）
  return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}
```

Dispatch key 的优先级顺序在 `c10/core/DispatchKeySet.h` 定义，从高到低（简化）：
```
BackendSelect → Python → PythonSnapshot → FuncTorchDynamicLayerBack →
Named → Conjugate → Negative → ZeroTensor →
ADInplaceOrView → AutogradXXX → Tracer →
AutocastCPU/CUDA → Batched → VmapMode →
Functionalize → CPU/CUDA/MPS/...
```

一次 `torch.add(cpu_a, cpu_b)`（在 no_grad 下）的实际 kernel 序列：
```
BackendSelect (跳过) → CPU: wrapper_CPU_add_Tensor
```

一次 `torch.add(cpu_a.requires_grad_(True), cpu_b)`：
```
AutogradCPU: VariableType::add_Tensor  →  (redispatch)  →  CPU: wrapper_CPU_add_Tensor
```

---

## 8. Autograd 层

### 8.1 求导规则：`derivatives.yaml`

`tools/autograd/derivatives.yaml:229`：

```yaml
- name: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  other: handle_r_to_c(other.scalar_type(), maybe_multiply(grad, alpha.conj()))
  result: self_t + maybe_multiply(other_t, alpha)      # 前向 AD (JVP)

- name: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  result: self_t.clone()
```

- `self:` / `other:` 是反向公式（每个 differentiable 输入一行）；`result:` 是前向 AD。
- `maybe_multiply(grad, alpha.conj())`：若 `alpha == 1` 会被优化掉。
- `handle_r_to_c`：处理 real → complex 的梯度类型转换。

### 8.2 生成 `AddBackward0` 与 `VariableType::add_Tensor`

生成器：`tools/autograd/gen_autograd_functions.py`（生成 `Functions.cpp`）+ `tools/autograd/gen_variable_type.py`（生成 `VariableType_*.cpp`）。

生成物形态（简化）：

```cpp
// torch/csrc/autograd/generated/Functions.h
struct TORCH_API AddBackward0 : public TraceableFunction {
  variable_list apply(variable_list&& grads) override;
  Scalar alpha;
  at::ScalarType self_scalar_type;
  at::ScalarType other_scalar_type;
};

// torch/csrc/autograd/generated/Functions.cpp
variable_list AddBackward0::apply(variable_list&& grads) {
  auto& grad = grads[0];
  variable_list grad_inputs(2);
  if (should_compute_output({ self_ix })) {
    grad_inputs[self_ix] = handle_r_to_c(self_scalar_type, grad);
  }
  if (should_compute_output({ other_ix })) {
    grad_inputs[other_ix] = handle_r_to_c(other_scalar_type,
                                          maybe_multiply(grad, alpha.conj()));
  }
  return grad_inputs;
}

// torch/csrc/autograd/generated/VariableType_2.cpp
at::Tensor add_Tensor(c10::DispatchKeySet ks,
                      const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto _any_requires_grad = compute_requires_grad(self, other);
  std::shared_ptr<AddBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::make_shared<AddBackward0>();
    grad_fn->set_next_edges(collect_next_edges(self, other));
    grad_fn->alpha = alpha;
    grad_fn->self_scalar_type  = self.scalar_type();
    grad_fn->other_scalar_type = other.scalar_type();
  }
  auto result = ([&]() {
    // 剥掉 Autograd 键，向下 redispatch → 走到 CPU/CUDA kernel
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::add_Tensor::redispatch(
      ks & c10::after_autograd_keyset, self, other, alpha);
  })();
  if (grad_fn) set_history(flatten_tensor_args(result), grad_fn);
  return result;
}
```

`gen_variable_type.py:895` 附近的 `wrapper_registrations` 生成：

```cpp
// VariableType_2.cpp (@generated)
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl("add.Tensor", TORCH_FN(VariableType::add_Tensor));
  m.impl("add_.Tensor", TORCH_FN(VariableType::add__Tensor));
  m.impl("add.out",   TORCH_FN(VariableType::add_out_out));
  m.impl("add.Scalar", TORCH_FN(VariableType::add_Scalar));
  ...
}
```

因为 `AutogradCPU`、`AutogradCUDA` 等继承自 `Autograd`，任何 backend 只要走 requires_grad 路径都会先命中它。

### 8.3 ADInplaceOrView 键

`ADInplaceOrViewType.cpp` 里为 `add_.Tensor` 注册版本号 bump 逻辑（inplace ops 的历史追踪）。

---

## 9. Python 绑定

### 9.1 `Tensor::add` 成员方法

生成到 `ATen/core/TensorBody.h`（模板：`aten/src/ATen/templates/TensorBody.h`）：

```cpp
inline at::Tensor Tensor::add(const at::Tensor& other, const at::Scalar& alpha) const {
  return at::_ops::add_Tensor::call(const_cast<Tensor&>(*this), other, alpha);
}
inline at::Tensor& Tensor::add_(const at::Tensor& other, const at::Scalar& alpha) const {
  return at::_ops::add__Tensor::call(const_cast<Tensor&>(*this), other, alpha);
}
inline at::Tensor Tensor::add(const at::Scalar& other, const at::Scalar& alpha) const {
  return at::_ops::add_Scalar::call(const_cast<Tensor&>(*this), other, alpha);
}
```

`yaml` 里 `variants: function, method` 决定生成哪些形式。

### 9.2 `torch.add` / `Tensor.add`（Python）

生成器：`tools/autograd/gen_python_functions.py`
模板：`tools/autograd/templates/python_torch_functions.cpp` / `python_variable_methods.cpp`

`torch.add` 走 `python_torch_functions.cpp` 中的 sharded 文件（生成物形态）：

```cpp
// torch::autograd
static PyObject* THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
    "add(Tensor input, Scalar other, Scalar alpha=1)",
  }, ...);
  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  ...
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // torch.add(Tensor, Tensor, alpha=)
        return wrap(dispatch_add(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // torch.add(..., out=)
        return wrap(dispatch_add_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
    }
    case 1: {
      return wrap(dispatch_add(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
    }
  }
  END_HANDLE_TH_ERRORS
}

static PyMethodDef torch_functions_shard[] = {
  {"add", castPyCFunctionWithKeywords(THPVariable_add), METH_VARARGS|METH_KEYWORDS, nullptr},
  ...
};
```

`dispatch_add` 是内部小 helper：`AutoNoGIL guard; return at::add(self, other, alpha);`——释放 GIL、进入 ATen 世界。

`Tensor.add` (bound method) 走 `python_variable_methods.cpp`，结构类似但第一个参数是 `THPVariable_Unpack(self_)`。

---

## 10. 端到端调用链

以 Python 侧 `c = torch.add(a, b)`（`a`, `b` 是需要梯度的 CUDA Tensor）为例：

```
Python: torch.add(a, b)
   │  (torch._C._VariableFunctions.add)
   ▼
python_torch_functions.cpp: THPVariable_add
   │  - PythonArgParser 解析
   │  - AutoNoGIL 释放 GIL
   ▼
ATen/ops/add.h: at::add(self, other, alpha)
   ▼
Operators_*.cpp: at::_ops::add_Tensor::call(self, other, alpha)
   │  - findSchemaOrThrow → TypedOperatorHandle
   ▼
Dispatcher::call
   │  - dispatchKeyExtractor → {AutogradCUDA, CUDA, ...}
   │  - lookup 命中最高键 AutogradCUDA
   ▼
VariableType_*.cpp: VariableType::add_Tensor  (Autograd wrapper)
   │  - 构造 AddBackward0，保存 alpha/dtypes
   │  - collect_next_edges(a, b)
   │  - AutoDispatchBelowADInplaceOrView { redispatch(ks & after_autograd_keyset) }
   ▼
Dispatcher::redispatch
   │  - lookup 命中 CUDA
   ▼
RegisterCUDA.cpp: wrapper_CUDA_add_Tensor
   │  - 实例化 structured_ufunc_add_CUDA_functional
   │  - op.meta(a, b, alpha)  →  TensorIterator::build
   │  - op.impl(a, b, alpha, out)
   ▼
UfuncCUDA_add.cu: TORCH_IMPL_FUNC(ufunc_add_CUDA)::impl
   │  - add_stub(kCUDA, iter, alpha)  (由 REGISTER_DISPATCH 挂钩)
   ▼
UfuncCUDA_add.cu: add_kernel_cuda(iter, alpha)
   │  - AT_DISPATCH_...  →  gpu_kernel(iter, CUDAFunctor_add<scalar_t>{alpha_})
   ▼
CUDA kernel: 每个线程调用 ufunc::add(a, b, alpha)  →  a + alpha * b
   │
   ▼
返回 out Tensor
   │
   ▲  set_history(result, grad_fn=AddBackward0)  ← 回到 VariableType 层
   │
   ▲  return  ← 回到 Python，wrap 成 THPVariable
```

反向：`c.sum().backward()` 时 `AddBackward0::apply({grad_c})` 被调度，返回 `{handle_r_to_c(dtype_a, grad_c), handle_r_to_c(dtype_b, maybe_multiply(grad_c, alpha.conj()))}`，各自累加进 `a.grad` / `b.grad`。

---

## 11. 想给 `add` 加一个新 backend 时要动哪些文件

假设要给虚构 backend `Foo`（`DispatchKey::Foo`）加 `add`：

1. **`native_functions.yaml`**：给 `add.out` 的 `dispatch:` 段加一行 `Foo: add_out_foo`（如果是结构化算子）或给 `add.Tensor` 加 `Foo: add_foo`。
2. **实现函数**：`aten/src/ATen/native/foo/BinaryOps.cpp` 里写 `TORCH_IMPL_FUNC(add_out_foo)` 或普通函数 `at::Tensor& add_out_foo(...)`。
3. **构建系统**：把新 cpp/cu 加入 `aten/src/ATen/CMakeLists.txt` 的 backend 源码列表；如果是新的 dispatch key，还要在 `c10/core/DispatchKey.h` 注册。
4. **不用**改 dispatcher/schema/自动求导/Python 绑定——codegen 会自动为 `Foo` 生成 `wrapper_Foo_add_out_out`，并写 `TORCH_LIBRARY_IMPL(aten, Foo, m) { m.impl("add.out", TORCH_FN(wrapper_Foo_add_out_out)); }` 到 `RegisterFoo.cpp`。
5. 如果 backend 需要特殊的 autograd 行为（大多数不需要），才需要在 `derivatives.yaml` 加 `dispatch:` 段并写 `Foo_add_Tensor` 反向。

对于新的**外部 backend**（在 PyTorch 外定义），可以在自己的库里写：

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", my_add_impl);
}
```

用 `DispatchKey::PrivateUse1` 复用现有的分派路径，不需要改 PyTorch 源码。

---

## 12. 关键文件速查表

| 类别 | 路径 |
| :--- | :--- |
| Schema 事实源 | `aten/src/ATen/native/native_functions.yaml` |
| 求导事实源 | `tools/autograd/derivatives.yaml` |
| Meta 函数（形状） | `aten/src/ATen/native/BinaryOps.cpp` (`TORCH_META_FUNC2(add, Tensor)`) |
| Dispatch stub 声明 | `aten/src/ATen/native/BinaryOps.h` |
| Ufunc 数学 | `aten/src/ATen/native/ufunc/add.h` |
| Ufunc codegen（生成器） | `torchgen/api/ufunc.py`, `torchgen/dest/ufunc.py` |
| Ufunc codegen（模板） | `aten/src/ATen/templates/UfuncCPU.cpp`, `UfuncCPUKernel.cpp`, `UfuncCUDA.cu` |
| Dispatch key wrapper 模板 | `aten/src/ATen/templates/RegisterDispatchKey.cpp`, `RegisterDispatchDefinitions.ini` |
| Dispatch key wrapper 生成器 | `torchgen/dest/register_dispatch_key.py`, `torchgen/gen.py` |
| Schema 注册模板 | `aten/src/ATen/templates/RegisterSchema.cpp` |
| Autograd wrapper 模板 | `tools/autograd/templates/VariableType.cpp`, `Functions.{h,cpp}` |
| Autograd wrapper 生成器 | `tools/autograd/gen_variable_type.py`, `gen_autograd_functions.py` |
| Python 绑定模板 | `tools/autograd/templates/python_torch_functions.cpp`, `python_variable_methods.cpp` |
| Python 绑定生成器 | `tools/autograd/gen_python_functions.py` |
| Dispatcher 核心 | `aten/src/ATen/core/dispatch/Dispatcher.h/.cpp`, `torch/library.h`, `aten/src/ATen/core/library.cpp` |
| 手写 kernel（举例） | `aten/src/ATen/native/mps/operations/BinaryOps.mm`, `.../sparse/SparseTensorMath.cpp`, `.../mkldnn/BinaryOps.cpp`, `.../nested/NestedTensorBinaryOps.cpp` |

生成产物（构建后可在 `build/aten/src/ATen/` 或安装的 wheel `torch/include/ATen/ops/` 找到）：

```
ATen/ops/add.h                    # 用户 API
ATen/ops/add_ops.h                # _ops::add_Tensor / add_out / add_Scalar ...
ATen/ops/add_native.h             # native::structured_ufunc_add_CPU / _CUDA / 手写 kernel 声明
ATen/ops/add_meta.h               # meta::structured_add_Tensor
ATen/ops/add_cpu_dispatch.h       # at::cpu::add(...)
ATen/ops/add_cuda_dispatch.h      # at::cuda::add(...)
ATen/ops/add_compositeexplicitautograd_dispatch.h    # add.Scalar
ATen/RegisterCPU.cpp              # TORCH_LIBRARY_IMPL(aten, CPU, m) { m.impl("add.out", ...); }
ATen/RegisterCUDA.cpp
ATen/RegisterMPS.cpp
ATen/RegisterSparseCPU.cpp
ATen/RegisterCompositeExplicitAutograd.cpp
ATen/RegisterSchema.cpp           # TORCH_LIBRARY(aten, m) { m.def(...); }
ATen/Operators_*.cpp              # add_Tensor::call → Dispatcher
ATen/UfuncCPU_add.cpp             # DEFINE_DISPATCH(add_stub); TORCH_IMPL_FUNC(ufunc_add_CPU)
ATen/UfuncCPUKernel_add.cpp       # add_kernel + REGISTER_DISPATCH(add_stub, ...)
ATen/UfuncCUDA_add.cu             # CUDAFunctor_add<...> + REGISTER_DISPATCH(add_stub, ...)
torch/csrc/autograd/generated/VariableType_2.cpp   # Autograd wrapper + TORCH_LIBRARY_IMPL(aten, Autograd, m)
torch/csrc/autograd/generated/Functions.cpp        # AddBackward0::apply
torch/csrc/autograd/generated/python_torch_functions_2.cpp  # THPVariable_add
```

---

## 13. 一句话总结

- **`native_functions.yaml` + `derivatives.yaml` + `ufunc/add.h`** 是**唯一手写**的东西（约 30 行）。
- 其余所有 —— schema 注册、Dispatcher 入口、backend wrapper、structured meta 类、autograd wrapper、backward node、Python 绑定 —— 都由 `torchgen/`（ATen）和 `tools/autograd/`（Autograd + Python）**在编译期批量生成**。
- 运行时靠 `c10::Dispatcher` 按 `DispatchKeySet` 做**分层派发**：`Autograd → BackendSelect → CPU/CUDA/MPS/Sparse/Nested/...`，每一层用 `TORCH_LIBRARY_IMPL(aten, KEY, m) { m.impl(...) }` 挂钩，链式 `redispatch` 向下走到实际 kernel。
