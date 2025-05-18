import os
import sys
import platform
import subprocess
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # 确保 cmake 可用
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake必须安装才能构建扩展")

        # 检查扩展模块
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # 构建目录
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            # 传递其他参数给 CMake
            "-DUSE_CUDA=ON",  # 默认启用CUDA
        ]

        # 配置模式
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        # 配置平台相关参数
        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
            build_args += ["--", "-j4"]

        # 创建构建目录
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # 执行CMake构建
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args + [f"-Dpybind11_DIR={ext.sourcedir}/third_party/pybind11"], cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

setup(
    name="ridoa",
    version="0.1.0",
    author="RIDOA Team",
    author_email="example@example.com",
    description="基于旋转干涉仪的多目标参数估计系统",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/ridoa",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    ext_modules=[CMakeExtension("ridoa.ridoa_core")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
    ],
)