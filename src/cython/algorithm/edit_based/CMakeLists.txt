
add_cython_target(Hamming CXX)
add_library(Hamming MODULE ${Hamming})
target_compile_features(Hamming PUBLIC cxx_std_14)
target_include_directories(Hamming PRIVATE
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/
    ${RAPIDFUZZ_CAPI_PATH}
    ${RAPIDFUZZ_BASE_DIR}/src/cython)
python_extension_module(Hamming)
install(TARGETS Hamming LIBRARY DESTINATION src/rapidfuzz/algorithm/edit_based)

add_cython_target(Levenshtein CXX)
add_library(Levenshtein MODULE ${Levenshtein})
target_compile_features(Levenshtein PUBLIC cxx_std_14)
target_include_directories(Levenshtein PRIVATE
    ${RAPIDFUZZ_BASE_DIR}/extern/rapidfuzz-cpp/
    ${RAPIDFUZZ_CAPI_PATH}
    ${RAPIDFUZZ_BASE_DIR}/src/cython)
python_extension_module(Levenshtein)
install(TARGETS Levenshtein LIBRARY DESTINATION src/rapidfuzz/algorithm/edit_based)
