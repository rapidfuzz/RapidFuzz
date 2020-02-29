#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "process.hpp"


PYBIND11_MODULE(rapidfuzz, m) {
    m.doc() = R"pbdoc(
        rapid string matching library
    )pbdoc";

    m.def("extract_one", &extract_one, R"pbdoc(
        Find the best match in a list of matches
    )pbdoc");

    m.attr("__version__") = VERSION_INFO;
}