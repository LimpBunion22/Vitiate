#!/bin/bash

cd ./utils/VIT-FPGA
make libs
cp ./include/netFPGA.h $HOME/workspace_development/include/
cp ./include/fpgaHandler.h $HOME/workspace_development/include/
cp ./include/kernelCore.h $HOME/workspace_development/include/
cp ./def/fpgaDefines.h $HOME/workspace_development/include/


cd ../../build

make build
make install

cd ..
conda activate Vitiate