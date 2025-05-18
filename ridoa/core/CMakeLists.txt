cmake_minimum_required(VERSION 3.12)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda
)

# 设置C++源文件
set(CXX_SOURCES 
    src/cpp/array_processing.cpp 
    src/cpp/doa_system.cpp 
    src/cpp/music_algorithm.cpp 
    src/cpp/parameter_space.cpp 
    src/cpp/phase_matrix.cpp
)

# 如果启用CUDA，添加CUDA源文件
if(USE_CUDA)
    set(CUDA_SOURCES  
        src/cuda/music_cuda.cu 
        src/cuda/parameter_cuda.cu
    )
    # 针对CUDA文件的特殊编译选项
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# 创建静态库
add_library(ridoa_core_static STATIC ${CXX_SOURCES} ${CUDA_SOURCES})
set_target_properties(ridoa_core_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON   # 添加PIC支持
)
target_include_directories(ridoa_core_static PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/eigen>
    $<INSTALL_INTERFACE:include/eigen>
)

# 如果使用CUDA，添加相关链接和定义
if(USE_CUDA)
    target_link_libraries(ridoa_core_static PRIVATE ${CUDA_LIBRARIES})
    target_compile_definitions(ridoa_core_static PRIVATE USE_CUDA)

    # 添加CUDA特定编译选项
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # 添加relaxed-constexpr选项，允许在设备代码中调用主机constexpr函数
    target_compile_options(ridoa_core_static PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# Python绑定部分
set(BINDING_SOURCES 
    bindings/pybind_module.cpp 
)

# 创建Python模块
pybind11_add_module(ridoa_core ${BINDING_SOURCES})
target_link_libraries(ridoa_core PRIVATE ridoa_core_static)

# 安装规则
install(TARGETS ridoa_core_static
    EXPORT RIDOATargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(TARGETS ridoa_core
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/ridoa
)

install(DIRECTORY include/
    DESTINATION include/ridoa
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)

# 导出目标
install(EXPORT RIDOATargets
    FILE RIDOATargets.cmake
    NAMESPACE RIDOA::
    DESTINATION lib/cmake/RIDOA
)