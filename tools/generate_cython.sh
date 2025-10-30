#!/bin/sh
curdir="${0%/*}/../src/rapidfuzz"

linetrace_flag=""
if [ "$1" = "--linetrace" ]; then
  linetrace_flag="-X linetrace=True"
  echo "Line tracing enabled"
fi

generate_cython()
{
  python -m cython -I "$curdir" --cplus $linetrace_flag "$curdir"/"$1".pyx -o "$curdir"/"$1".cxx || exit 1
  echo "Generated $curdir/$1.cxx"
}

generate_cython fuzz_cpp
generate_cython fuzz_cpp_avx2
generate_cython fuzz_cpp_sse2
generate_cython process_cpp_impl
generate_cython utils_cpp
generate_cython _feature_detector_cpp

generate_cython distance/_initialize_cpp
generate_cython distance/metrics_cpp
generate_cython distance/metrics_cpp_avx2
generate_cython distance/metrics_cpp_sse2
