# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100)
endif()

project(foundation_pose_nvidia_wide_requirements
  VERSION 0.1.0
  DESCRIPTION "Production-oriented NVIDIA FoundationPose implementation"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(PNG REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT 10 root directory")
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS "${TENSORRT_ROOT}" ENV TENSORRT_ROOT
  PATH_SUFFIXES include)
find_library(TENSORRT_INFER_LIBRARY
  NAMES nvinfer nvinfer_10
  HINTS "${TENSORRT_ROOT}" ENV TENSORRT_ROOT
  PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
find_library(TENSORRT_ONNXPARSER_LIBRARY
  NAMES nvonnxparser nvonnxparser_10
  HINTS "${TENSORRT_ROOT}" ENV TENSORRT_ROOT
  PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_INFER_LIBRARY OR NOT TENSORRT_ONNXPARSER_LIBRARY)
  message(FATAL_ERROR
    "TensorRT 10 headers/libraries are required. Set -DTENSORRT_ROOT=/path/to/TensorRT "
    "or install libnvinfer-dev and libnvonnxparsers-dev.")
endif()

# Optional nvdiffrast-backed renderer. The main library always ships the
# dependency-free built-in CUDA rasterizer (the default renderer). When this
# option is ON, the nvdiffrast CUDA-rasterizer core (NVIDIA Source Code
# License; requires an NVlabs/nvdiffrast checkout fetched separately) is
# compiled into its OWN shared library, libfoundation_pose_nvdiffrast.so,
# which the main library loads dynamically (dlopen) only when the runtime
# selection FP_RENDERER=nvdiffrast is requested. The main library never
# contains or links nvdiffrast code, keeping the license boundary at the
# shared-library boundary.
option(FOUNDATION_POSE_WITH_NVDIFFRAST "Build the optional nvdiffrast renderer plugin" OFF)

add_library(foundation_pose_nvidia SHARED
  src/builtin_raster_renderer.cu
  src/c_api.cpp
  src/cuda_pipeline.cu
  src/device_buffer.cpp
  src/foundation_pose.cpp
  src/mesh.cpp
  src/pipeline_utils.cpp
  src/tensorrt_runner.cu)

target_include_directories(foundation_pose_nvidia
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${TENSORRT_INCLUDE_DIR})

target_compile_definitions(foundation_pose_nvidia
  PRIVATE FOUNDATION_POSE_NVIDIA_EXPORTS)

# Find the sibling renderer plugin next to the main library at runtime even
# without LD_LIBRARY_PATH.
#
# SOVERSION is the C ABI contract for c_api.h. Bump it whenever a public struct
# (e.g. fp_config_t) grows or a public signature changes, so a client built
# against an older header fails to load instead of silently reading/writing
# past its smaller struct.
set_target_properties(foundation_pose_nvidia PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN")

if(FOUNDATION_POSE_WITH_NVDIFFRAST)
  set(NVDIFFRAST_ROOT "" CACHE PATH "Path to an NVlabs/nvdiffrast checkout")
  if(NOT NVDIFFRAST_ROOT)
    message(FATAL_ERROR "Set -DNVDIFFRAST_ROOT=/path/to/nvdiffrast (or -DFOUNDATION_POSE_WITH_NVDIFFRAST=OFF)")
  endif()
  if(NOT EXISTS "${NVDIFFRAST_ROOT}/csrc/common/rasterize.cu")
    message(FATAL_ERROR "NVDIFFRAST_ROOT must point to an NVlabs/nvdiffrast checkout with csrc/common/rasterize.cu")
  endif()

  set(NVDIFFRAST_COMMON_SOURCES
    "${NVDIFFRAST_ROOT}/csrc/common/common.cpp"
    "${NVDIFFRAST_ROOT}/csrc/common/cudaraster/impl/Buffer.cpp"
    "${NVDIFFRAST_ROOT}/csrc/common/cudaraster/impl/CudaRaster.cpp"
    "${NVDIFFRAST_ROOT}/csrc/common/cudaraster/impl/RasterImpl.cpp"
    "${NVDIFFRAST_ROOT}/csrc/common/cudaraster/impl/RasterImpl_kernel.cu"
    "${NVDIFFRAST_ROOT}/csrc/common/rasterize.cu")

  add_library(foundation_pose_nvdiffrast SHARED
    src/nvdiffrast_renderer.cu
    src/device_buffer.cpp
    ${NVDIFFRAST_COMMON_SOURCES})
  target_include_directories(foundation_pose_nvdiffrast
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      "${NVDIFFRAST_ROOT}/csrc")
  target_compile_definitions(foundation_pose_nvdiffrast PRIVATE NVDR_STANDALONE)
  target_link_libraries(foundation_pose_nvdiffrast PRIVATE CUDA::cudart)
  target_compile_options(foundation_pose_nvdiffrast PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
  install(TARGETS foundation_pose_nvdiffrast LIBRARY DESTINATION lib)
endif()

target_link_libraries(foundation_pose_nvidia
  PRIVATE
    CUDA::cudart
    ${CMAKE_DL_LIBS}
    PNG::PNG
    ${TENSORRT_INFER_LIBRARY}
    ${TENSORRT_ONNXPARSER_LIBRARY})

target_compile_options(foundation_pose_nvidia PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wpedantic>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

add_executable(foundation_pose_nvidia_cli tools/foundation_pose_cli.cpp)
target_link_libraries(foundation_pose_nvidia_cli PRIVATE foundation_pose_nvidia)

install(TARGETS foundation_pose_nvidia foundation_pose_nvidia_cli
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
