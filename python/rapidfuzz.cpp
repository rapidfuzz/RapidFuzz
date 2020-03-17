#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "process.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rapidfuzz, m) {
    m.doc() = R"pbdoc(
        rapid string matching library
    )pbdoc";

    auto mprocess = m.def_submodule("process");
 
    mprocess.def("extractOne", &extract_one,
                 py::arg("query"), py::arg("choices"), py::arg("score_cutoff") = 0
                 R"pbdoc(
                    Find the best match in a list of matches
                 )pbdoc");

    m.attr("__version__") = VERSION_INFO;
}
