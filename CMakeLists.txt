cmake_minimum_required(VERSION 3.15.0)

option(WITH_FPGA "build with FPGA" ON)

if(WITH_FPGA)
    add_compile_definitions(USE_FPGA)
    message(STATUS "using FPGA")
else()
    message(STATUS "FPGA disabled")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# choose compiler
set(CMAKE_C_COMPILER "$ENV{HOME}/intel/oneapi/compiler/latest/linux/bin/icx")
set(CMAKE_CXX_COMPILER "$ENV{HOME}/intel/oneapi/compiler/latest/linux/bin/icpx")
if(WITH_FPGA)
    set(AOCL_COMPILE_FLAGS "-fstack-protector -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fPIE -fPIC")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${AOCL_COMPILE_FLAGS}")
    set(AOCL_LINK_FLAGS "-L/home/hai/intelFPGA_pro/18.1/hld/host/linux64/lib -z noexecstack -Wl,-z,relro,-z,now -Wl,-Bsymbolic -pie -lOpenCL")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${AOCL_LINK_FLAGS} -lstdc++fs")
endif()

project(netStandalone VERSION 0.1.0)

#dirs
set(pybind11_DIR "$ENV{HOME}/anaconda3/envs/Vitiate/share/cmake/pybind11")
set(PYTHON_LIBRARY_DIR "$ENV{HOME}/anaconda3/envs/Vitiate/lib/python3.8/site-packages")
set(PYTHON_EXECUTABLE "$ENV{HOME}/anaconda3/envs/Vitiate/bin/python")
set(LIB_FOLDER "$ENV{HOME}/workspace_development/lib")

# #packages
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

#dirs paths
include_directories("/usr/local/cuda/include") #INCLUDE THIS FIRST BECAUSE IT SHARES HEADERS WITH FPGA, AND CUDA OPENCL HAS MORE DECLARATIONS?
include_directories("def")
include_directories("include")
include_directories("$ENV{HOME}/workspace_development/include")
if(WITH_FPGA)
    include_directories("$ENV{HOME}/intelFPGA_pro/18.1/hld/host/include")
    include_directories("$ENV{HOME}/intelFPGA_pro/18.1/hld/examples_aoc/common/inc")
endif()

#files to include
file (GLOB SRC_FILES "src/*.cpp")
file (GLOB DEF_FILES "def/*.h")
file (GLOB INCLUDE_FILES "include/*.h")

list(APPEND CMAKE_INSTALL_RPATH "$ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin")
list(APPEND CMAKE_INSTALL_RPATH "/usr/local/cuda/lib64")
if(WITH_FPGA)
    list(APPEND CMAKE_INSTALL_RPATH "$ENV{HOME}/intelFPGA_pro/18.1/hld/host/linux64/lib")
endif()

# #pybind 
pybind11_add_module(netStandalone
    ${DEF_FILES}
    ${SRC_FILES}
    ${INCLUDE_FILES})

# # executable
# add_executable(netStandalone
#     ${DEF_FILES}
#     ${SRC_FILES}
#     ${INCLUDE_FILES})

set_target_properties(netStandalone PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(netStandalone PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(netStandalone PRIVATE
    ${OpenCV_LIBS}
    $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libsvml.so
    $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libintlc.so.5
    $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libirng.so   
    $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libimf.so
    /usr/local/cuda/lib64/libcudart.so
    /usr/local/cuda/lib64/libcublas.so
    ${LIB_FOLDER}/libnetCPU.so
    ${LIB_FOLDER}/libnetGPU.so)

if(WITH_FPGA)
    target_link_libraries(netStandalone PRIVATE 
        stdc++fs
        $ENV{HOME}/intelFPGA_pro/18.1/hld/host/linux64/lib/libOpenCL.so
        ${LIB_FOLDER}/libnetFPGA.a)
endif()

#install to conda python lib dir
install(TARGETS netStandalone DESTINATION ${PYTHON_LIBRARY_DIR})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
