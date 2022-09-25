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
Python_add_library(utils_cpp MODULE WITH_SOABI ${utils_cpp} ${CMAKE_CURRENT_LIST_DIR}/utils.cpp)
target_compile_features(utils_cpp PUBLIC cxx_std_17)
target_include_directories(utils_cpp PRIVATE ${RF_CAPI_PATH} ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(utils_cpp PRIVATE rapidfuzz::rapidfuzz)
install(TARGETS utils_cpp LIBRARY DESTINATION src/rapidfuzz)

create_cython_target(fuzz_cpp)
Python_add_library(fuzz_cpp MODULE WITH_SOABI ${fuzz_cpp})
target_compile_features(fuzz_cpp PUBLIC cxx_std_17)
target_include_directories(fuzz_cpp PRIVATE ${RF_CAPI_PATH} ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(fuzz_cpp PRIVATE rapidfuzz::rapidfuzz)
install(TARGETS fuzz_cpp LIBRARY DESTINATION src/rapidfuzz)

create_cython_target(process_cpp_impl)
Python_add_library(process_cpp_impl MODULE WITH_SOABI ${process_cpp_impl})
target_compile_features(process_cpp_impl PUBLIC cxx_std_17)
target_include_directories(process_cpp_impl PRIVATE ${RF_CAPI_PATH} ${RF_BASE_DIR}/rapidfuzz)
target_link_libraries(process_cpp_impl PRIVATE Taskflow::Taskflow rapidfuzz::rapidfuzz)

# check if we need to link with libatomic (not needed on Windows)
if (NOT Windows)
	# TODO: migrate to CheckSourceCompiles in CMake >= 3.19
	include(CheckCXXSourceCompiles)

	set(ATOMICS_TEST_SOURCE [=[
		#include <atomic>
		#include <cstdint>
		std::atomic<int> x{0};
		int main() {
			x.fetch_add(1, std::memory_order_relaxed);
			return 0;
		}
	]=])
	string(REPLACE "std::atomic<int>" "std::atomic<std::int8_t>" ATOMICS8_TEST_SOURCE "${ATOMICS_TEST_SOURCE}")
	string(REPLACE "std::atomic<int>" "std::atomic<std::int64_t>" ATOMICS64_TEST_SOURCE "${ATOMICS_TEST_SOURCE}")

	if(APPLE)
		set(CMAKE_REQUIRED_FLAGS "-std=c++11")
	endif()
	check_cxx_source_compiles("${ATOMICS_TEST_SOURCE}" HAVE_CXX_ATOMICS_WITHOUT_LIB)
	check_cxx_source_compiles("${ATOMICS8_TEST_SOURCE}" HAVE_CXX_ATOMICS8_WITHOUT_LIB)
	check_cxx_source_compiles("${ATOMICS64_TEST_SOURCE}" HAVE_CXX_ATOMICS64_WITHOUT_LIB)
	if((NOT HAVE_CXX_ATOMICS_WITHOUT_LIB) OR (NOT HAVE_CXX_ATOMICS8_WITHOUT_LIB) OR (NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB))
		set(CMAKE_REQUIRED_LIBRARIES "atomic")
		check_cxx_source_compiles("${ATOMICS_TEST_SOURCE}" HAVE_CXX_ATOMICS_WITH_LIB)
		check_cxx_source_compiles("${ATOMICS8_TEST_SOURCE}" HAVE_CXX_ATOMICS8_WITH_LIB)
		check_cxx_source_compiles("${ATOMICS64_TEST_SOURCE}" HAVE_CXX_ATOMICS64_WITH_LIB)
		if ((NOT HAVE_CXX_ATOMICS_WITH_LIB) OR (NOT HAVE_CXX_ATOMICS8_WITH_LIB) OR (NOT HAVE_CXX_ATOMICS64_WITH_LIB))
			message(FATAL_ERROR "No native support for std::atomic, or libatomic not found!")
		else()
			message(STATUS "Linking with libatomic for atomics support")
			unset(CMAKE_REQUIRED_LIBRARIES)
			target_link_libraries(process_cpp_impl PUBLIC atomic)
		endif()
	endif()
	if(APPLE)
		unset(CMAKE_REQUIRED_FLAGS)
	endif()
endif()

install(TARGETS process_cpp_impl LIBRARY DESTINATION src/rapidfuzz)
