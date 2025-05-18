#include "../types.h"
#include <vector>

Eigen::MatrixXd ParameterSpaceCUDA(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart, double elevationEnd,
    double azimuthStart, double azimuthEnd,
    double precision);