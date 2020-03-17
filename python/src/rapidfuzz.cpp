#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "process.hpp"
#include "fuzz.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rapidfuzz, m) {
    m.doc() = R"pbdoc(
        rapid string matching library
    )pbdoc";

    auto mprocess = m.def_submodule("process");

    mprocess.def("extract", &process::extract,
                 py::arg("query"), py::arg("choices"), py::arg("score_cutoff") = 0,
                 R"pbdoc(Find all matches with a ratio above score_cutoff)pbdoc");

    mprocess.def("extractOne", &process::extract_one,
                 py::arg("query"), py::arg("choices"), py::arg("score_cutoff") = 0,
                 R"pbdoc(Find the best match in a list of matches)pbdoc");


    auto mfuzz = m.def_submodule("fuzz");
    mfuzz.def("ratio", &fuzz::ratio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);

    mfuzz.def("QRatio", &fuzz::QRatio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);

    mfuzz.def("WRatio", &fuzz::WRatio,
              py::arg("a"), py::arg("b"), py::arg("score_cutoff") = 0);

    m.attr("__version__") = VERSION_INFO;
}
