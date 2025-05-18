// include/hip/parameter_hip.hip
#pragma once

#include "../types.h"
#include <vector>

Eigen::MatrixXd ParameterSpaceHIP(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart, double elevationEnd,
    double azimuthStart, double azimuthEnd,
    double precision);