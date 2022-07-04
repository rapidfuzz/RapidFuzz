function(create_cython_target _name)
    if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/${_name}.cxx)
        set(${_name} ${CMAKE_CURRENT_LIST_DIR}/${_name}.cxx PARENT_SCOPE)
    else()
        find_package(Cython REQUIRED)
        # should use target_include_directories once this is supported by scikit-build
        include_directories(${RF_BASE_DIR}/rapidfuzz)
        add_cython_target(${_name} CXX)
        set(${_name} ${_name} PARENT_SCOPE)
    endif()
endfunction(create_cython_target)

create_cython_target(utils_cpp)
add_library(utils_cpp MODULE ${utils_cpp} ${CMAKE_CURRENT_LIST_DIR}/utils.cpp)
target_compile_features(utils_cpp PUBLIC cxx_std_14)
target_include_directories(utils_cpp PRIVATE ${RF_CAPI_PATH} ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(utils_cpp PRIVATE rapidfuzz::rapidfuzz)
python_extension_module(utils_cpp)
install(TARGETS utils_cpp LIBRARY DESTINATION src/rapidfuzz)

create_cython_target(fuzz_cpp)
add_library(fuzz_cpp MODULE ${fuzz_cpp})
target_compile_features(fuzz_cpp PUBLIC cxx_std_14)
target_include_directories(fuzz_cpp PRIVATE ${RF_CAPI_PATH} ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(fuzz_cpp PRIVATE rapidfuzz::rapidfuzz)
python_extension_module(fuzz_cpp)
install(TARGETS fuzz_cpp LIBRARY DESTINATION src/rapidfuzz)

create_cython_target(string_metric_cpp)
add_library(string_metric_cpp MODULE ${string_metric_cpp})
target_compile_features(string_metric_cpp PUBLIC cxx_std_14)
target_include_directories(string_metric_cpp PRIVATE ${RF_CAPI_PATH} ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(string_metric_cpp PRIVATE rapidfuzz::rapidfuzz jaro_winkler::jaro_winkler)
python_extension_module(string_metric_cpp)
install(TARGETS string_metric_cpp LIBRARY DESTINATION src/rapidfuzz)

create_cython_target(process_cpp)
add_library(process_cpp MODULE ${process_cpp})
target_compile_features(process_cpp PUBLIC cxx_std_14)
target_include_directories(process_cpp PRIVATE ${RF_CAPI_PATH} ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(process_cpp PRIVATE rapidfuzz::rapidfuzz)
python_extension_module(process_cpp)
install(TARGETS process_cpp LIBRARY DESTINATION src/rapidfuzz)

create_cython_target(process_cdist_cpp)
add_library(process_cdist_cpp MODULE ${process_cdist_cpp})
target_compile_features(process_cdist_cpp PUBLIC cxx_std_14)
target_include_directories(process_cdist_cpp PRIVATE ${RF_CAPI_PATH} ${RF_BASE_DIR}/rapidfuzz ${NumPy_INCLUDE_DIR})
target_link_libraries(process_cdist_cpp PRIVATE Taskflow::Taskflow rapidfuzz::rapidfuzz)
python_extension_module(process_cdist_cpp)
install(TARGETS process_cdist_cpp LIBRARY DESTINATION src/rapidfuzz)