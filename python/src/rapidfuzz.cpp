#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "process.hpp"
#include "fuzz.hpp"
#include "utils.hpp"

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
        R"pbdoc(
            calculates a simple ratio between two strings

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): floating point betwee
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100

            Example:
                >>> fuzz.ratio("this is a test", "this is a test!")
                96.55171966552734
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);
    
    mfuzz.def("partial_ratio", &fuzz::partial_ratio,
        R"pbdoc(
            calculates a partial ratio between two strings

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100

            Example:
                >>> fuzz.partial_ratio("this is a test", "this is a test!")
                100.0
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);
    
    mfuzz.def("token_sort_ratio", &fuzz::token_sort_ratio,
        R"pbdoc(
            sorts the words in the string and calculates the fuzz.ratio between them

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100

            Example:
                >>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
                100.0
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);


    mfuzz.def("partial_token_sort_ratio", &fuzz::partial_token_sort_ratio,
        R"pbdoc(
            sorts the words in the strings and calculates the fuzz.partial_ratio between them

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);
    
    mfuzz.def("token_set_ratio", &fuzz::token_set_ratio,
        R"pbdoc(
            Compares the words in the strings based on unique and common words between them using fuzz.ratio

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100

            Example:
                >>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
                83.8709716796875
                >>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
                100.0
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);
    
    mfuzz.def("partial_token_set_ratio", &fuzz::partial_token_set_ratio,
        R"pbdoc(
            Compares the words in the strings based on unique and common words between them using fuzz.partial_ratio

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);

    mfuzz.def("token_ratio", &fuzz::token_ratio,
        R"pbdoc(
            Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
            (faster than manually executing the two functions)

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);
    
    mfuzz.def("partial_token_ratio", &fuzz::partial_token_ratio,
        R"pbdoc(
            Helper method that returns the maximum of fuzz.partial_token_set_ratio and fuzz.partial_token_sort_ratio
            (faster than manually executing the two functions)

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);

    mfuzz.def("WRatio", &fuzz::WRatio,
        R"pbdoc(
            Calculates a weighted ratio based on the other ratio algorithms

            Args:
                s1 (str): first string to compare
                s2 (str): second string to compare
                score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
                    For ratio < score_cutoff 0 is returned instead. Defaults to 0.
                preprocess (bool): Optional argument to specify whether the strings should be preprocessed 
                    using utils.default_process. Defaults to True.
            
            Returns:
                float: ratio between s1 and s2 as a float between 0 and 100
        )pbdoc",
        py::arg("s1"), py::arg("s2"), py::arg("score_cutoff") = 0, py::arg("preprocess") = true);
    

    auto mutils = m.def_submodule("utils");
    mutils.def("default_process", &utils::default_process,
        R"pbdoc(
        )pbdoc");

    m.attr("__version__") = VERSION_INFO;
}
