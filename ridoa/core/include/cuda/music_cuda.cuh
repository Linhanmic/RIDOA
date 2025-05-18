#include <vector>
#include "../types.h"

std::tuple<std::vector<double>, std::vector<double>>
computeMusicSpectrumCUDA(const ComplexMatrix &noiseSubspaceProduct,
                         double startAngle, double endAngle, double stepAngle);