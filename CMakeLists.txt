cmake_minimum_required(VERSION 3.12)

project(RIDOA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# CUDA配置
option(USE_CUDA "Use CUDA acceleration" ON)
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    
    # 检测CUDA版本和可用架构
    include(CheckLanguage)
    check_language(CUDA)
    
    # 设置CUDA架构
    # 可根据目标硬件调整，下面设置支持较广泛的架构
    set(CMAKE_CUDA_ARCHITECTURES 86)
    
    # 添加CUDA编译定义
    add_definitions(-DUSE_CUDA)
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