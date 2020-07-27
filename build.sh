#!/bin/bash

# Build libfacedetection
cd ./Detectors/libfacedetection
rm -r build
mkdir -p build
cd ./build

myarch=$(uname -m)
echo "Your arch is $myarch"
if [ $myarch = "x86_64" ]
then
    cmake_args="-DCMAKE_INSTALL_PREFIX=install -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DDEMO=OFF -DENABLE_AVX2=ON -DENABLE_NEON=OFF"
fi

if [ $myarch = "aarch64" ]
then
    cmake_args="-DCMAKE_INSTALL_PREFIX=install -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DDEMO=OFF -DENABLE_AVX2=OFF -DENABLE_NEON=ON"
fi 

cmake .. $cmake_args
cmake --build . --config Release