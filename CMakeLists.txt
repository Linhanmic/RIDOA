# 修改后的CMakeLists.txt以支持HIP编译
cmake_minimum_required(VERSION 3.12)

project(RIDOA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HIP REQUIRED)
if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
if(HIP_PLATFORM STREQUAL "amd")
    set(USE_HIP ON)
    message(STATUS "HIP_PLATFORM: ${HIP_PLATFORM}")
    find_package(hip REQUIRED)
    set(HIP_LIBRARIES hip::host)
elseif(HIP_PLATFORM STREQUAL "nvidia")
    set(USE_CUDA ON)
    message(STATUS "HIP_PLATFORM: ${HIP_PLATFORM}")
    enable_language(CUDA)
    # 使用现代CMake CUDA支持，不再使用已弃用的FindCUDA模块
    find_package(CUDAToolkit REQUIRED)
    set(HIP_LIBRARIES CUDA::cudart)
    set(HIP_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
endif()

# 硬件加速选项配置
option(USE_CUDA "Use CUDA acceleration" ON)
option(USE_HIP "Use HIP acceleration for AMD GPUs" OFF)

# 只能选择一种硬件加速方式
if(USE_CUDA AND USE_HIP)
    message(WARNING "同时启用CUDA和HIP可能会导致问题，优先使用HIP")
    set(USE_CUDA OFF)
endif()

# CUDA配置
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # 设置CUDA架构
    set(CMAKE_CUDA_ARCHITECTURES 86)
    
    # 添加CUDA编译定义
    add_definitions(-DUSE_CUDA)
endif()

# HIP配置
if(USE_HIP)
    #  添加HIP编译定义
    add_definitions(-DUSE_HIP)
endif()

# 设置安装目录 (如果未指定，默认为系统目录)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory" FORCE)
endif()

# 禁用LTO，防止链接问题
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# 设置构建类型 (如果未指定)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# 输出构建信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA enabled: ${USE_CUDA}")
message(STATUS "HIP enabled: ${USE_HIP}")
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")

# 添加第三方库
add_subdirectory(third_party/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
add_subdirectory(third_party/eigen ${CMAKE_CURRENT_BINARY_DIR}/eigen)

# 添加子项目
add_subdirectory(ridoa/core ${CMAKE_CURRENT_BINARY_DIR}/ridoa_core)

# 安装脚本和文档
install(FILES 
    README.md
    DESTINATION share/ridoa)

# 生成配置文件
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RIDOAConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/RIDOAConfig.cmake"
    @ONLY
)

# 安装配置文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/RIDOAConfig.cmake"
    DESTINATION lib/cmake/RIDOA
)