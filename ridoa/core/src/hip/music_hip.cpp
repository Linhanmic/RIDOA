// src/hip/music_hip.cpp
#include "hip/hip_helper.hip"
#include "hip/music_hip.hip"
#include "doa_config.h"

// HIP核函数：计算MUSIC谱
__global__ void computeSpectrumKernelHIP(hipDoubleComplex *noiseSubspaceProduct,
                                         double startAngle,
                                         double stepAngle,
                                         int numElements,
                                         double elementSpacing,
                                         double *spectrum,
                                         int numAngles,
                                         int matrixSize)
{
    // 使用共享内存存储导向矢量
    extern __shared__ hipDoubleComplex sharedMem[];
    hipDoubleComplex *steeringVector = sharedMem;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAngles)
        return;

    double angle = startAngle + idx * stepAngle;
    angle = angle * M_PI / 180.0; // 转换为弧度

    // 预先计算sin(angle)以减少重复计算
    double sinAngle = sin(angle);

    // 计算该方向导向矢量并存储在共享内存中
    for (int e = 0; e < numElements; ++e)
    {
        // elementSpacing已经以半波长为单位，所以这里需要 * 0.5 * 2π = π
        double phase = M_PI * elementSpacing * e * sinAngle;
        steeringVector[threadIdx.x * numElements + e] = make_hipDoubleComplex(cos(phase), sin(phase));
    }

    // 确保所有线程都完成了共享内存的写入
    __syncthreads();

    // 计算谱值 P = 1 / (a^H * U_n * U_n^H * a)
    hipDoubleComplex result = make_hipDoubleComplex(0.0, 0.0);

    for (int i = 0; i < numElements; ++i)
    {
        hipDoubleComplex tempRow = make_hipDoubleComplex(0.0, 0.0);
        for (int j = 0; j < numElements; ++j)
        {
            int index = i * numElements + j;
            if (index < matrixSize)
            {
                tempRow = hipCadd(tempRow,
                                  hipCmul(noiseSubspaceProduct[index],
                                          steeringVector[threadIdx.x * numElements + j]));
            }
        }
        result = hipCadd(result,
                         hipCmul(hipConj(steeringVector[threadIdx.x * numElements + i]), tempRow));
    }

    // MUSIC算法取倒数
    double power;
    if (hipCabs(result) > 1e-10)
    {
        power = 1.0 / hipCabs(result);
    }
    else
    {
        power = 1e10; // 避免除以零
    }

    // 保存结果
    spectrum[idx] = power;
}

// 执行MUSIC算法
std::tuple<std::vector<double>, std::vector<double>> computeMusicSpectrumHIP(
    const ComplexMatrix &noiseSubspaceProduct,
    double startAngle,
    double endAngle,
    double stepAngle)
{
    // 检查是否有可用的HIP设备
    if (!initializeHIP(DOAConfig::getInstance().gpuDeviceId))
    {
        throw std::runtime_error("无法初始化HIP设备，无法执行GPU加速");
    }

    int numAngles = static_cast<int>((endAngle - startAngle) / stepAngle) + 1;
    std::vector<double> spectrum(numAngles);
    std::vector<double> angles(numAngles);

    // 如果使用了空间平滑，噪声子空间投影矩阵的大小会小于阵元数目
    int nElements = DOAConfig::getInstance().nElements;
    if (DOAConfig::getInstance().forwardSmoothingSize > 0)
    {
        nElements = DOAConfig::getInstance().forwardSmoothingSize;
    }

    // 计算角度
    for (int i = 0; i < numAngles; ++i)
    {
        angles[i] = startAngle + i * stepAngle;
    }

    // 准备HIP内存
    hipDoubleComplex *d_noiseSubspaceProduct = nullptr;
    double *d_spectrum = nullptr;

    // 分配HIP内存
    size_t noiseMatrixSize = noiseSubspaceProduct.rows() * noiseSubspaceProduct.cols() * sizeof(hipDoubleComplex);
    CHECK_HIP_ERROR(hipMalloc(&d_noiseSubspaceProduct, noiseMatrixSize));
    CHECK_HIP_ERROR(hipMalloc(&d_spectrum, numAngles * sizeof(double)));

    // 将Eigen矩阵转换为线性数组, 按行存储
    std::vector<hipDoubleComplex> hostNoiseMatrix(noiseSubspaceProduct.size());
    for (int i = 0; i < noiseSubspaceProduct.rows(); ++i)
    {
        for (int j = 0; j < noiseSubspaceProduct.cols(); ++j)
        {
            ComplexType value = noiseSubspaceProduct(i, j);
            hostNoiseMatrix[i * noiseSubspaceProduct.cols() + j] =
                make_hipDoubleComplex(value.real(), value.imag());
        }
    }

    // 复制数据到GPU
    CHECK_HIP_ERROR(hipMemcpy(d_noiseSubspaceProduct, hostNoiseMatrix.data(),
                              noiseMatrixSize, hipMemcpyHostToDevice));

    // 执行HIP核函数
    int blockSize = 256;
    int numBlocks = (numAngles + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * nElements * sizeof(hipDoubleComplex);

    hipLaunchKernelGGL(computeSpectrumKernelHIP,
                       dim3(numBlocks), dim3(blockSize), sharedMemSize, 0,
                       d_noiseSubspaceProduct, startAngle, stepAngle,
                       nElements, DOAConfig::getInstance().elementSpacing, d_spectrum, numAngles,
                       noiseSubspaceProduct.size());

    // 检查HIP核函数执行是否有错误
    CHECK_HIP_ERROR(hipGetLastError());
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // 复制结果回主机
    CHECK_HIP_ERROR(hipMemcpy(spectrum.data(), d_spectrum,
                              numAngles * sizeof(double), hipMemcpyDeviceToHost));

    // 释放HIP内存
    CHECK_HIP_ERROR(hipFree(d_noiseSubspaceProduct));
    CHECK_HIP_ERROR(hipFree(d_spectrum));

    return {spectrum, angles};
}