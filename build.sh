#!/bin/bash
# 自动构建脚本 - build.sh

set -e

# 默认值
BUILD_TYPE="Release"
USE_CUDA="ON"
INSTALL_PREFIX="./install"
BUILD_DIR="./build"
JOBS=4

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-cuda)
      USE_CUDA="OFF"
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
      echo "  --no-cuda        禁用CUDA支持"
      echo "  --debug          构建Debug版本"
      echo "  --prefix=DIR     设置安装目录"
      echo "  --jobs=N         设置并行构建任务数"
      echo "  --clean          清理构建目录"
      echo "  --help           显示此帮助信息"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      echo "使用 $0 --help 获取帮助"
      exit 1
      ;;
  esac
done

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置
echo "配置 RIDOA 项目..."
echo "  构建类型: $BUILD_TYPE"
echo "  CUDA支持: $USE_CUDA"
echo "  安装目录: $INSTALL_PREFIX"

cmake .. \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DUSE_CUDA="$USE_CUDA" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

# 构建
echo "编译项目..."
cmake --build . --config "$BUILD_TYPE" --parallel "$JOBS"

# 安装
echo "安装到 $INSTALL_PREFIX..."
cmake --install .

echo "构建完成!"
echo "RIDOA 已安装到: $INSTALL_PREFIX"