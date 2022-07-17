#!/bin/sh
curdir="${0%/*}"

generate_cython()
{
  python -m cython -I "$curdir" --cplus "$curdir"/"$1".pyx -o "$curdir"/"$1".cxx || exit 1
  echo "Generated $curdir/$1.cxx"
}

generate_cython fuzz_cpp
generate_cython process_cdist_cpp_impl
generate_cython process_cpp
generate_cython string_metric_cpp
generate_cython utils_cpp

generate_cython distance/_initialize_cpp
generate_cython distance/Hamming_cpp
generate_cython distance/Indel_cpp
generate_cython distance/LCSseq_cpp
generate_cython distance/Levenshtein_cpp
