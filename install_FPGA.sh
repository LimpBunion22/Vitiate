#!/bin/bash

cd ./utils/VIT-FPGA
make
cp ./include/netFPGA.h $HOME/workspace_development/include/

cd ../../build

make build
make install

cd ..
conda activate Vitiate