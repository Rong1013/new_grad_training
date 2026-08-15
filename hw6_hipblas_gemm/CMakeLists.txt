# hw6 GEMM demo：hipBLAS + 手写 kernel。
# 用 hipcc 直接当 C++ 编译器；ROCm/DTK 提供了 hip 和 hipblas 的 CMake package。
cmake_minimum_required(VERSION 3.18)

# 用 dcc 或 hipcc 当编译器。命令行传 -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc 也行。
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/opt/dtk/bin/hipcc" CACHE FILEPATH "" FORCE)
endif()

project(hw6_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 让 CMake 找到 DTK 里的 hip/hipblas 包
list(APPEND CMAKE_PREFIX_PATH "/opt/dtk" "/opt/dtk/hip" "/opt/dtk/hipblas")

# 编译期开优化 + 目标架构
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# 目标 GPU 架构。gfx936 是 DCU-K100AI，K100 是 gfx928，Z100 是 gfx906。
# 想同时编多个，用分号分隔。
if(NOT DEFINED GPU_ARCHS)
  set(GPU_ARCHS "gfx936" CACHE STRING "target GPU arch(es), separated by ;")
endif()

add_compile_options(-O3 -Wall -Wno-unused-result)
foreach(arch ${GPU_ARCHS})
  add_compile_options(--offload-arch=${arch})
  add_link_options(--offload-arch=${arch})
endforeach()

# 目标 1：hipBLAS SGEMM
add_executable(hipblas_gemm src/hipblas_gemm.cpp)
target_include_directories(hipblas_gemm PRIVATE /opt/dtk/include)
target_link_directories(hipblas_gemm PRIVATE /opt/dtk/lib)
target_link_libraries(hipblas_gemm PRIVATE hipblas amdhip64)

# 目标 2：手写 tiled SGEMM
add_executable(custom_gemm src/custom_gemm.cpp)
target_include_directories(custom_gemm PRIVATE /opt/dtk/include)
target_link_directories(custom_gemm PRIVATE /opt/dtk/lib)
target_link_libraries(custom_gemm PRIVATE amdhip64)

# 一个方便运行的自定义 target：make run
add_custom_target(run
  COMMAND $<TARGET_FILE:hipblas_gemm> 1024 1024 1024 30
  COMMAND $<TARGET_FILE:custom_gemm>  1024 1024 1024 30
  DEPENDS hipblas_gemm custom_gemm
  USES_TERMINAL
)

# 一个 hipprof 采集 target：make profile
add_custom_target(profile
  COMMAND /opt/dtk/bin/hipprof --hip-trace --output-type 0
          -o ${CMAKE_BINARY_DIR}/hipblas_trace $<TARGET_FILE:hipblas_gemm> 1024 1024 1024 30
  COMMAND /opt/dtk/bin/hipprof --hip-trace --output-type 0
          -o ${CMAKE_BINARY_DIR}/custom_trace $<TARGET_FILE:custom_gemm> 1024 1024 1024 30
  DEPENDS hipblas_gemm custom_gemm
  USES_TERMINAL
)
