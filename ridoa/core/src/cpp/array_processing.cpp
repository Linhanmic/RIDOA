// src/cpp/array_processing.cpp
#include "array_processing.h"
#include "phase_matrix.h"
#include <random>
#include <stdexcept>

ArrayProcessor::ArrayProcessor()
{
}

ComplexMatrix ArrayProcessor::generateSimulationData(
    const std::vector<double> &elevations,
    const std::vector<double> &azimuths,
    double duration,
    double snrDb) const
{

    int numSignals = elevations.size();
    if (numSignals != azimuths.size())
    {
        throw std::invalid_argument("俯仰角和方位角数组长度必须相同");
    }

    // 计算信噪比线性值
    double snrLinear = std::pow(10.0, snrDb / 10.0);
    double noisePower = 1.0;
    double signalPower = snrLinear * noisePower;
    double signalAmplitude = std::sqrt(signalPower);
    double noiseAmplitude = std::sqrt(noisePower);

    // 生成采样点数
    int numSamples = static_cast<int>(duration * config_.samplingRate);

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disPhaseDist(0, 2 * M_PI);
    std::normal_distribution<> disNoiseDist(0, 1);

    // 生成数据矩阵
    ComplexMatrix dataMatrix(config_.nElements, numSamples);

    // 初始化相位差矩阵计算器
    PhaseMatrix phaseMatrixCalculator;

    // 生成每个采样点的数据
    for (int i = 0; i < numSamples; ++i)
    {
        double t = static_cast<double>(i) / config_.samplingRate;

        // 生成信号
        std::vector<ComplexType> signals(numSignals);
        for (int j = 0; j < numSignals; ++j)
        {
            double initialPhase = disPhaseDist(gen);
            double phase = 2 * M_PI * config_.carrierFrequency * t + initialPhase;
            signals[j] = signalAmplitude * std::exp(ComplexType(0, phase));
        }

        // 计算阵元相位差矩阵
        ComplexMatrix phaseDiff = phaseMatrixCalculator.compute(t, elevations, azimuths);

        // 计算阵元接收信号
        Eigen::VectorXcd dataVector = Eigen::VectorXcd::Zero(config_.nElements);
        for (int e = 0; e < config_.nElements; ++e)
        {
            for (int j = 0; j < numSignals; ++j)
            {
                dataVector[e] += phaseDiff(e, j) * signals[j];
            }
        }

        // 添加噪声
        for (int e = 0; e < config_.nElements; ++e)
        {
            ComplexType noise(noiseAmplitude * disNoiseDist(gen),
                              noiseAmplitude * disNoiseDist(gen));
            dataVector[e] += noise;
        }

        // 将数据加入矩阵
        dataMatrix.col(i) = dataVector;
    }

    return dataMatrix;
}

ComplexMatrix ArrayProcessor::forwardSpatialSmoothing(
    const ComplexMatrix &dataMatrix,
    int subArraySize) const
{

    if (subArraySize <= 0 || subArraySize > config_.nElements)
    {
        // 不进行平滑处理，直接返回原始协方差矩阵
        return dataMatrix * dataMatrix.adjoint() / static_cast<double>(dataMatrix.cols());
    }

    int numSubArrays = config_.nElements - subArraySize + 1;
    if (numSubArrays <= 0)
    {
        throw std::runtime_error("子阵大小设置不正确，无法形成有效子阵");
    }

    ComplexMatrix smoothedCov = ComplexMatrix::Zero(subArraySize, subArraySize);

    // 对每个子阵计算协方差矩阵并累加
    for (int i = 0; i < numSubArrays; ++i)
    {
        ComplexMatrix subArray = dataMatrix.block(i, 0, subArraySize, dataMatrix.cols());
        smoothedCov += subArray * subArray.adjoint();
    }

    // 归一化
    smoothedCov /= static_cast<double>(numSubArrays * dataMatrix.cols());

    return smoothedCov;
}

int ArrayProcessor::estimateNumSources(
    const Eigen::VectorXd &eigenvalues,
    double threshold) const
{

    // 使用能量比例法
    double totalEnergy = eigenvalues.sum();
    double cumulativeEnergy = 0.0;

    for (int i = 0; i < eigenvalues.size(); ++i)
    {
        cumulativeEnergy += eigenvalues(i);
        if (cumulativeEnergy / totalEnergy >= threshold)
        {
            return i + 1;
        }
    }

    return 1; // 默认至少有一个信号源
}

ComplexMatrix ArrayProcessor::computeNoiseSubspace(
    const ComplexMatrix &dataMatrix,
    int numSignals) const
{

    // 执行前向空间平滑处理
    int subArraySize = (config_.forwardSmoothingSize > 0) ? config_.forwardSmoothingSize : config_.nElements;

    ComplexMatrix covarianceMatrix = forwardSpatialSmoothing(dataMatrix, subArraySize);

    // 确保协方差矩阵是Hermitian的
    covarianceMatrix = (covarianceMatrix + covarianceMatrix.adjoint()) * 0.5;

    // 计算特征值和特征向量
    Eigen::SelfAdjointEigenSolver<ComplexMatrix> eigensolver(covarianceMatrix);
    if (eigensolver.info() != Eigen::Success)
    {
        throw std::runtime_error("特征值分解失败");
    }

    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues().reverse(); // 从大到小排序
    ComplexMatrix eigenvectors = eigensolver.eigenvectors();

    // 反转特征向量顺序以匹配特征值顺序
    ComplexMatrix sortedEigenvectors(eigenvectors.rows(), eigenvectors.cols());
    for (int i = 0; i < eigenvectors.cols(); ++i)
    {
        sortedEigenvectors.col(i) = eigenvectors.col(eigenvectors.cols() - 1 - i);
    }

    // 如果未指定信号源数量，则自动估计
    if (numSignals <= 0)
    {
        numSignals = estimateNumSources(eigenvalues);
        numSignals = std::min(numSignals, config_.maxSources); // 限制最大数量
    }

    // 计算噪声子空间
    int noiseSubspaceDim = sortedEigenvectors.cols() - numSignals;
    if (noiseSubspaceDim <= 0)
    {
        throw std::runtime_error("噪声子空间维度必须为正");
    }

    ComplexMatrix noiseEigenvectors = sortedEigenvectors.rightCols(noiseSubspaceDim);

    // 返回噪声子空间投影矩阵 U_n * U_n^H
    return noiseEigenvectors * noiseEigenvectors.adjoint();
}