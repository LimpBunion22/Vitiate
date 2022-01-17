#!/bin/bash

cd ./utils/VIT-FPGA
source install_VIT_FPGA.sh

cd ../../build
export LD_LIBRARY_PATH=$HOME/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$HOME/intelFPGA_pro/18.1/hld/host/linux64/lib:$LD_LIBRARY_PATH
make build
make install
cd ..

# for file in "$include"/*
# do
#     cp "$include"/"$file" $HOME/workspace_development/include/
# done