#!/bin/bash
# 自动构建脚本 - build.sh (修改以支持HIP)

set -e

# 默认值
BUILD_TYPE="Release"
ACCELERATOR="auto"  # 自动选择CUDA或HIP
INSTALL_PREFIX="./install"
BUILD_DIR="./build"
JOBS=4

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --cpu-only)
      ACCELERATOR="cpu"
      shift
      ;;
    --cuda)
      ACCELERATOR="cuda"
      shift
      ;;
    --hip)
      ACCELERATOR="hip"
      shift
      ;;
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --prefix=*)
      INSTALL_PREFIX="${1#*=}"
      shift
      ;;
    --jobs=*)
      JOBS="${1#*=}"
      shift
      ;;
    --clean)
      echo "清理构建目录..."
      rm -rf "$BUILD_DIR"
      echo "构建目录 $BUILD_DIR 已清理"
      exit 0
      ;;
    --help)
      echo "RIDOA 构建脚本"
      echo "用法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --cpu-only        禁用GPU加速，仅使用CPU"
      echo "  --cuda            使用CUDA加速 (NVIDIA GPU)"
      echo "  --hip             使用HIP加速 (AMD GPU)"
      echo "  --debug           构建Debug版本"
      echo "  --prefix=DIR      设置安装目录"
      echo "  --jobs=N          设置并行构建任务数"
      echo "  --clean           清理构建目录"
      echo "  --help            显示此帮助信息"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      echo "使用 $0 --help 获取帮助"
      exit 1
      ;;
  esac
done

# 检测可用的GPU加速类型
detect_gpu_type() {
  # 首先检查是否有NVIDIA GPU
  if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU，优先使用CUDA"
    echo "cuda"
    return
  fi

  # 然后检查是否有AMD GPU
  if command -v rocm-smi &> /dev/null || [ -d "/opt/rocm" ]; then
    echo "检测到AMD GPU，使用HIP"
    echo "hip"
    return
  fi

  # 如果都没有检测到，使用CPU
  echo "没有检测到支持的GPU，使用CPU计算"
  echo "cpu"
}

# 如果选择自动检测加速器类型
if [ "$ACCELERATOR" = "auto" ]; then
  ACCELERATOR=$(detect_gpu_type)
fi

# 设置CMake参数
CMAKE_ARGS=()
case $ACCELERATOR in
  "cuda")
    CMAKE_ARGS+=("-DUSE_CUDA=ON" "-DUSE_HIP=OFF")
    ;;
  "hip")
    CMAKE_ARGS+=("-DUSE_CUDA=OFF" "-DUSE_HIP=ON")
    ;;
  "cpu")
    CMAKE_ARGS+=("-DUSE_CUDA=OFF" "-DUSE_HIP=OFF")
    ;;
esac

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置
echo "配置 RIDOA 项目..."
echo "  构建类型: $BUILD_TYPE"
echo "  加速器: $ACCELERATOR"
echo "  安装目录: $INSTALL_PREFIX"

cmake .. \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  "${CMAKE_ARGS[@]}" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

# 构建
echo "编译项目..."
cmake --build . --config "$BUILD_TYPE" --parallel "$JOBS"

# 安装
echo "安装到 $INSTALL_PREFIX..."
cmake --install .

echo "构建完成!"
echo "RIDOA 已安装到: $INSTALL_PREFIX"