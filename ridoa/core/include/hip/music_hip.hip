// include/hip/music_hip.hip
#pragma once

#include <vector>
#include "../types.h"

std::tuple<std::vector<double>, std::vector<double>>
computeMusicSpectrumHIP(const ComplexMatrix &noiseSubspaceProduct,
                         double startAngle, double endAngle, double stepAngle);