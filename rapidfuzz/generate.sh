#!/bin/sh
curdir="${0%/*}"

generate_cython()
{
  python -m cython -I "$curdir" --cplus "$curdir"/"$1".pyx -o "$curdir"/"$1".cxx || exit 1
  echo "Generated $curdir/$1.cxx"
}

generate_cython cpp_fuzz
generate_cython cpp_process_cdist
generate_cython cpp_process
generate_cython cpp_string_metric
generate_cython cpp_utils

generate_cython distance/_initialize
generate_cython distance/Hamming
generate_cython distance/Indel
generate_cython distance/Jaro
generate_cython distance/JaroWinkler
generate_cython distance/Levenshtein
