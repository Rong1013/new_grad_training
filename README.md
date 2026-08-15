# 应届生培养计划

学习资料地址：svn://42.228.13.241/DCUAI/部门培训

## 公司文化、账号、环境（2天）
蓉光研发资源使用指南

## AI 计算基础与硬件认知（4天）
### 学习内容：
- CPU、GPU、NPU 的基本差异。
- GPU 执行模型：线程、线程块、warp、SM、显存层级。
- AI 加速器常见概念：Tensor Core、HBM、DMA、算子融合。
- 框架、算子库、驱动、硬件之间的调用关系。
### 资料
- DCU 架构
- hip-runtime与gpufusion：
svn://42.228.13.241/DCUAI/部门培训/02-cuda相关资料
svn://42.228.13.241/DCUAI/部门培训/03-HIP编程相关资料
https://download.sourcefind.cn:65024/1/main/latest/Document
### 作业
1. 一张框架到硬件的调用链路图
2. 简单的reduce cuda算子（功能正常、精度fp32满足双万分之一）

## 数学库与算子库基础（1天）
### 学习内容（了解）：
- BLAS 基础：GEMM、GEMV
- 常见数学库：rocBLAS。
- 常见深度学习算子库：cuDNN、MIOpen。
- 常见算子：MatMul、Conv、BatchNorm、LayerNorm、Softmax、Attention。
- 数据布局：NCHW、NHWC、row-major、column-major。
- 精度类型：FP32、FP16、BF16、INT8。
### 资料
svn://42.228.13.241/DCUAI/部门培训/07-部门技术培训资料/3-DTK软件栈
### 作业
1. 用 PyTorch 调用矩阵乘法、LayerNorm，并用 profiler 观察底层调用；

## AI 框架整体架构（5天）
### 学习内容：
- AI 框架分层：
  - Python API / 前端接口
  - Tensor 抽象
  - Operator 算子系统
  - Autograd 自动求导
  - Graph / IR 计算图
  - Runtime 运行时
  - Backend 后端适配
- 动态图与静态图差异。
- PyTorch、TensorFlow、Paddle 的架构对比。
- 框架中的 dispatch 机制：根据设备、dtype、layout 选择实现。
### 资料
svn://42.228.13.241/DCUAI/部门培训/07-部门技术培训资料/4-AI框架基础及进阶
pytorch官方文档：https://docs.pytorch.org/docs/2.12/index.html
### 作业：
1. 阅读 PyTorch 中Operator、Kernel 注册相关代码。
2. 跟踪一次 torch.add() 算子的执行路径。
3. 画出一个算子从 Python API 到后端 kernel 的调用链。（交付）

## PyTorch 如何适配支持的nvidia-GPU、amd-GPU、NPU（5天）
### 学习内容：
- PyTorch 支持一个新硬件需要改哪些层。
- Tensor、Device、DispatchKey、Operator Kernel 之间的关系。
- 新硬件如何通过 PyTorch Dispatcher 接管算子执行。
- 一个算子如何从 torch.add(x, y) 路由到自定义硬件 kernel。
### 资料
1. pytorch官方文档：https://docs.pytorch.org/docs/2.12/index.html
2. pytorch源码；
### 作业
1. PyTorch 中 operator 和 kernel 有什么区别？
2. DispatchKey 是什么？为什么新硬件需要它？
3. tensor.to("cuda") 背后需要哪些 runtime 能力？
4. 新硬件适配为什么必须考虑 allocator、stream、event？
5. 完成一个pytorch版本 dcu 上的编译。

## 模型性能瓶颈分析（3天）
### 学习内容：
- 使用 torch.profiler 拉取模型性能数据。
- 看懂 profiler 输出中的 Self CPU、CPU total、CUDA time、# of Calls、Input Shapes。
- 能区分 CPU 瓶颈、GPU瓶颈、数据加载瓶颈、框架调度瓶颈。
- 能导出 trace.json，使用 Chrome tracing 或 TensorBoard 查看时间线。
- 能针对一个模型输出性能分析报告。
### 资料
https://docs.pytorch.ac.cn/tutorials/recipes/recipes/profiler_recipe.html
- table怎么看：
- trace性能瓶颈现象：
### 作业
1. 跑以下教学模型，导出trace数据，找出3处优化点；
python3 train_perf_bottleneck_model.py --profile
2. 分析下面BERT4Rec模型profile性能数据，给出2点性能优化建议；

## 计算图、IR、编译优化（5天）
### 学习内容：
- 计算图表示方式。
- Graph IR / MLIR / ONNX 基础概念。
- 常见图优化：
  - constant folding
  - dead code elimination
  - operator fusion
  - layout transform
- eager mode、graph mode、AOT 编译。
### 资料
https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler.html
### 作业
- 对比 eager 执行和 torch.compile 执行性能，为什么会有性能提升。
- 分析一个模型中哪些算子之间可以融合，又哪些之间不能融;

## 基于Pytorch生态介绍runtime (3天）
### 学习内容：
1. Hip-runtime: 【开发环境使用手册】【HIP最佳实践手册】
2. gpufusion（cuda)： 【CUDA程序适配手册】
3. hip/rocm 与 cuda的差异: 熟悉基础类型/接口差异，基础库差异；
4. 熟悉DTK下的库与NV下库的对应关系；
5. Core Dump分析：Core Dump 分析
6. 性能分析工具：03-8-HIP程序性能分析工具
### 资料：
1. dtk相关资料汇总：https://download.sourcefind.cn:65024/1/main/latest/Document
### 作业：
1. 使用hipblas实现一个gemm计算demo;
2. 参考最佳实践手册实现一个手写gemm kernel;
3. 了解cmake的基本用法；
4. 使用hipprof做一次性能分析；


## 生态组件编译 （2天）
### 学习内容：
1. pytorch扩展编译支持(CUDAExtension/CppExtension/pybind)；
https://docs.pytorch.ac.cn/tutorials/advanced/cpp_extension.html
2. 基于torch的组件应用编译方法,
3. FastPT工具介绍以及使用；
### 作业
1. 完成一个组件应用的适配；组件分配待定
