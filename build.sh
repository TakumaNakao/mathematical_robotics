#!/bin/bash

cd example
cmake -S ./ -B build
cmake --build build -j || exit 1
cp -f build/point_cloud_matching ../