#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
build_dir=${DIR}/libs_build

git clone --depth 1 --branch llvmorg-11.0.0 https://github.com/llvm/llvm-project
cd llvm-project/openmp
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX="${build_dir}" -DCMAKE_MACOSX_RPATH="${build_dir}/lib" ..
make
make install

cd ../..
rm -rf llvm-project