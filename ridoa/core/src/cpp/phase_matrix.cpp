// src/cpp/phase_matrix.cpp
#include "phase_matrix.h"
#include <cmath>

PhaseMatrix::PhaseMatrix()
{
}

ComplexMatrix PhaseMatrix::compute(double time,
                                   const std::vector<double> &elevations,
                                   const std::vector<double> &azimuths) const
{
    int numElements = config_.nElements;
    int numSignals = elevations.size();
    ComplexMatrix phaseMatrix(numElements, numSignals);

    for (int e = 0; e < numElements; ++e)
    {
        for (int j = 0; j < numSignals; ++j)
        {
            // 转换为弧度
            double elevationRad = elevations[j] * M_PI / 180.0;
            double azimuthRad = azimuths[j] * M_PI / 180.0;

            // 相位差计算: φ(t) = 2π(d/λ)sinβcos(ωt-α)
            // 注意：elementSpacing已经以半波长为单位，所以这里需要 * 0.5 * 2π = π
            double phase = M_PI * config_.elementSpacing * e *
                           std::sin(elevationRad) * std::cos(config_.omega * time - azimuthRad);
            phaseMatrix(e, j) = std::exp(ComplexType(0, phase));
        }
    }

    return phaseMatrix;
}