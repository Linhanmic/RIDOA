// include/doa_result.h
#pragma once

#include <vector>
#include "types.h"

/**
 * @brief DOA估计结果类
 */
struct DOAResult
{
    std::vector<double> estElevations;    // 估计的俯仰角(度)
    std::vector<double> estAzimuths;      // 估计的方位角(度)
    double timepoint;                     // 时间点
    RealMatrix accumulator;               // 参数空间累加器(可选)
    std::vector<double> elevations;       // 参数空间俯仰角数组(可选)
    std::vector<double> azimuths;         // 参数空间方位角数组(可选)
    std::vector<double> angles;           // 一维模糊角度数组(可选)
    std::vector<double> anglesTimepoints; // 一维模糊角度对应时间点数组(可选)

    DOAResult() = default;
    DOAResult(const std::vector<double> &el,
              const std::vector<double> &az,
              double tp)
        : estElevations(el), estAzimuths(az), timepoint(tp) {}
};
