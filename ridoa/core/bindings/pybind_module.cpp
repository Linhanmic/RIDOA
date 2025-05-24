// bindings/pybind_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include "doa_system.h"
#include "doa_config.h"
#include "doa_result.h"

namespace py = pybind11;

using ComplexMatrix = Eigen::MatrixXcd;

void receiveSignalData(const ComplexMatrix &signalData)
{
    std::cout << "接收到信号数据：" << std::endl;
    std::cout << "行数: " << signalData.rows() << std::endl;
    std::cout << "列数: " << signalData.cols() << std::endl;
    std::cout << "示例数据: " << signalData(0, 0) << std::endl;
}

PYBIND11_MODULE(ridoa_core, m)
{
    m.doc() = "基于旋转干涉仪的多目标参数估计系统核心模块";

    // 导出DOAConfig类
    py::class_<DOAConfig>(m, "DOAConfig")
        .def_property("nElements", [](const DOAConfig &c)
                      { return c.nElements; }, [](DOAConfig &c, int v)
                      { c.nElements = v; })
        .def_property("elementSpacing", [](const DOAConfig &c)
                      { return c.elementSpacing; }, [](DOAConfig &c, double v)
                      { c.elementSpacing = v; })
        .def_property("omega", [](const DOAConfig &c)
                      { return c.omega; }, [](DOAConfig &c, double v)
                      { c.omega = v; })
        .def_property("samplingRate", [](const DOAConfig &c)
                      { return c.samplingRate; }, [](DOAConfig &c, int v)
                      { c.samplingRate = v; })
        .def_property("carrierFrequency", [](const DOAConfig &c)
                      { return c.carrierFrequency; }, [](DOAConfig &c, double v)
                      { c.carrierFrequency = v; })
        .def_property("estimateRate", [](const DOAConfig &c)
                      { return c.estimateRate; }, [](DOAConfig &c, int v)
                      { c.estimateRate = v; })
        .def_property("thetaPrecision", [](const DOAConfig &c)
                      { return c.thetaPrecision; }, [](DOAConfig &c, double v)
                      { c.thetaPrecision = v; })
        .def_property("precision", [](const DOAConfig &c)
                      { return c.precision; }, [](DOAConfig &c, double v)
                      { c.precision = v; })
        .def_property("accumulatorThreshold", [](const DOAConfig &c)
                      { return c.accumulatorThreshold; }, [](DOAConfig &c, double v)
                      { c.accumulatorThreshold = v; })
        .def_property("spectrumThreshold", [](const DOAConfig &c)
                      { return c.spectrumThreshold; }, [](DOAConfig &c, double v)
                      { c.spectrumThreshold = v; })
        .def_property("maxSources", [](const DOAConfig &c)
                      { return c.maxSources; }, [](DOAConfig &c, int v)
                      { c.maxSources = v; })
        .def_property("forwardSmoothingSize", [](const DOAConfig &c)
                      { return c.forwardSmoothingSize; }, [](DOAConfig &c, int v)
                      { c.forwardSmoothingSize = v; })
        .def_property("useGPU", [](const DOAConfig &c)
                      { return c.useGPU; }, [](DOAConfig &c, bool v)
                      { c.useGPU = v; })
        .def_property("gpuDeviceId", [](const DOAConfig &c)
                      { return c.gpuDeviceId; }, [](DOAConfig &c, int v)
                      { c.gpuDeviceId = v; })
        .def("print", &DOAConfig::print)
        .def_static("get_instance", &DOAConfig::getInstance, py::return_value_policy::reference);

    // 导出DOAResult类
    py::class_<DOAResult>(m, "DOAResult")
        .def(py::init<>())
        .def(py::init<const std::vector<double> &, const std::vector<double> &, double>())
        .def_readwrite("estElevations", &DOAResult::estElevations)
        .def_readwrite("estAzimuths", &DOAResult::estAzimuths)
        .def_readwrite("timepoint", &DOAResult::timepoint)
        .def_readwrite("accumulator", &DOAResult::accumulator)
        .def_readwrite("elevations", &DOAResult::elevations)
        .def_readwrite("azimuths", &DOAResult::azimuths)
        .def_readwrite("angles", &DOAResult::angles)
        .def_readwrite("anglesTimepoints", &DOAResult::anglesTimepoints);

    // 导出DOASystem类
    py::class_<DOASystem>(m, "DOASystem")
        .def(py::init<>())
        .def("generate_simulation_data", &DOASystem::generateSimulationData)
        .def("estimate_doa", &DOASystem::estimateDOA,
             py::arg("signal_data"), py::arg("num_signals") = -1);

    // 导出接收信号数据的函数
    // 导出接收信号数据的函数
    m.def("receive_signal_data", &receiveSignalData,
          "处理复数信号数据矩阵",
          py::arg("signalData"));
}