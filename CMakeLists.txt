cmake_minimum_required(VERSION 3.18.0)

option(WITH_BIND "build with binding" ON)
option(WITH_FPGA "build with FPGA" OFF)
option(WITH_GPU "build with GPU" ON)
option(WITH_OpenCV "build with OpenCV" ON)

if(WITH_FPGA)
    add_compile_definitions(USE_FPGA)
    message(STATUS "using FPGA")
else()
    message(STATUS "FPGA disabled")
endif()

if(WITH_BIND)
    add_compile_definitions(BIND)
    message(STATUS "builing for python")
else()
    message(STATUS "building exe")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# choose compiler
set(CMAKE_C_COMPILER "$ENV{HOME}/intel/oneapi/compiler/latest/bin/icx")
set(CMAKE_CXX_COMPILER "$ENV{HOME}/intel/oneapi/compiler/latest/bin/icpx")
if(WITH_FPGA)
    set(AOCL_COMPILE_FLAGS "-fstack-protector -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fPIE -fPIC")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${AOCL_COMPILE_FLAGS}")
    set(AOCL_LINK_FLAGS "-L/home/hai/intelFPGA_pro/18.1/hld/host/linux64/lib -z noexecstack -Wl,-z,relro,-z,now -Wl,-Bsymbolic -pie -lOpenCL")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${AOCL_LINK_FLAGS} -lstdc++fs")
endif()

project(netStandalone VERSION 0.1.0)

#dirs
# set(pybind11_DIR "$ENV{HOME}/anaconda3/envs/Vitiate/share/cmake/pybind11")
set(pybind11_DIR "vit/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
# set(PYTHON_LIBRARY_DIR "$ENV{HOME}/anaconda3/envs/Vitiate/lib/python3.8/site-packages")
set(PYTHON_LIBRARY_DIR "${CMAKE_BINARY_DIR}/../vit/lib/python3.12/site-packages")
# set(PYTHON_EXECUTABLE "$ENV{HOME}/anaconda3/envs/Vitiate/bin/python")
set(PYTHON_EXECUTABLE "vit/bin/python3")
set(LIB_FOLDER "$ENV{HOME}/workspace_development/lib")
# set(LIB_FOLDER "python/libs")

# #packages
if(WITH_OpenCV)
    find_package(OpenCV REQUIRED)
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

#dirs paths
if(WITH_GPU)
    include_directories("/usr/local/cuda/include") #INCLUDE THIS FIRST BECAUSE IT SHARES HEADERS WITH FPGA, AND CUDA OPENCL HAS MORE DECLARATIONS?
endif()
include_directories("def")
include_directories("include")
include_directories("$ENV{HOME}/workspace_development/include")
include_directories(${OpenCV_INCLUDE_DIRS})
if(WITH_FPGA)
    include_directories("$ENV{HOME}/intelFPGA_pro/18.1/hld/host/include")
    include_directories("$ENV{HOME}/intelFPGA_pro/18.1/hld/examples_aoc/common/inc")
endif()

#files to include
file (GLOB SRC_FILES "src/*.cpp")
file (GLOB DEF_FILES "def/*.h")
file (GLOB INCLUDE_FILES "include/*.h")

list(APPEND CMAKE_INSTALL_RPATH "$ENV{HOME}/intel/oneapi/compiler/latest/compiler/lib/intel64_lin")
if(WITH_GPU)
    list(APPEND CMAKE_INSTALL_RPATH "/usr/local/cuda/lib64")
endif()
if(WITH_FPGA)
    list(APPEND CMAKE_INSTALL_RPATH "$ENV{HOME}/intelFPGA_pro/18.1/hld/host/linux64/lib")
endif()

if(WITH_BIND)
#pybind 
pybind11_add_module(netStandalone
    ${DEF_FILES}
    ${SRC_FILES}
    ${INCLUDE_FILES})
else()
# executable
add_executable(netStandalone
    ${DEF_FILES}
    ${SRC_FILES}
    ${INCLUDE_FILES})
endif()

set_target_properties(netStandalone PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(netStandalone PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# target_link_libraries(netStandalone PRIVATE
#     ${OpenCV_LIBS}
#     $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libsvml.so
#     $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libintlc.so.5
#     $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libirng.so   
#     $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libimf.so
#     /usr/local/cuda/lib64/libcudart.so
#     /usr/local/cuda/lib64/libcublas.so
#     ${LIB_FOLDER}/libbuilderCPU.so
#     ${LIB_FOLDER}/libbuilderGPU.so)

list(APPEND LIBS 
    ${LIB_FOLDER}/libbuilderCPU.so
    $ENV{HOME}/intel/oneapi/compiler/latest/lib/libsvml.so
    $ENV{HOME}/intel/oneapi/compiler/latest/lib/libintlc.so.5
    $ENV{HOME}/intel/oneapi/compiler/latest/lib/libirng.so   
    $ENV{HOME}/intel/oneapi/compiler/latest/lib/libimf.so)

if(WITH_OpenCV)
    list(APPEND LIBS ${OpenCV_LIBS})
endif()
if(WITH_GPU)
    list(APPEND LIBS 
        /usr/local/cuda/lib64/libcudart.so 
        /usr/local/cuda/lib64/libcublas.so
        ${LIB_FOLDER}/libbuilderGPU.so)
endif()
if(WITH_FPGA)
    list(APPEND LIBS 
        stdc++fs
        $ENV{HOME}/intelFPGA_pro/18.1/hld/host/linux64/lib/libOpenCL.so
        ${LIB_FOLDER}/libnetFPGA.a)
endif()

target_link_libraries(netStandalone PRIVATE ${LIBS})

# if(WITH_FPGA)
#     target_link_libraries(netStandalone PRIVATE 
#         stdc++fs
#         $ENV{HOME}/intelFPGA_pro/18.1/hld/host/linux64/lib/libOpenCL.so
#         ${LIB_FOLDER}/libnetFPGA.a)
# endif()

if(WITH_BIND)
    #install to python env lib dir
    install(TARGETS netStandalone DESTINATION ${PYTHON_LIBRARY_DIR})
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
