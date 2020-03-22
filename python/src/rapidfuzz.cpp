#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "process.hpp"
#include "fuzz.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_rapidfuzz_cpp, m) {
    m.doc() = R"pbdoc(
        rapid string matching library
    )pbdoc";

    auto mprocess = m.def_submodule("process");
    mprocess.def("extract", &process::extract);
    mprocess.def("extractOne", &process::extractOne);

    auto mfuzz = m.def_submodule("fuzz");
    mfuzz.def("ratio", &fuzz::ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);
    
    mfuzz.def("partial_ratio", &fuzz::partial_ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);
    
    mfuzz.def("token_sort_ratio", &fuzz::token_sort_ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);
    
    mfuzz.def("partial_token_sort_ratio", &fuzz::partial_token_sort_ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);
    
    mfuzz.def("token_set_ratio", &fuzz::token_set_ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);
    
    mfuzz.def("partial_token_set_ratio", &fuzz::partial_token_set_ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);

    mfuzz.def("token_ratio", &fuzz::token_ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);
    
    mfuzz.def("partial_token_ratio", &fuzz::partial_token_ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);

    mfuzz.def("QRatio", &fuzz::QRatio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);

    mfuzz.def("WRatio", &fuzz::WRatio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);

    m.attr("__version__") = VERSION_INFO;
}
