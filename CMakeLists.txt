cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#choose compiler
set(CMAKE_C_COMPILER "$ENV{HOME}/intel/oneapi/compiler/latest/linux/bin/icx")
set(CMAKE_CXX_COMPILER "$ENV{HOME}/intel/oneapi/compiler/latest/linux/bin/icpx")

project(netStandalone VERSION 0.1.0)

#dirs
set(pybind11_DIR "$ENV{HOME}/anaconda3/envs/Vitiate/share/cmake/pybind11")
set(PYTHON_LIBRARY_DIR "$ENV{HOME}/anaconda3/envs/Vitiate/lib/python3.8/site-packages")
set(PYTHON_EXECUTABLE "$ENV{HOME}/anaconda3/envs/Vitiate/bin/python")
set(LIB_FOLDER "$ENV{HOME}/workspace_development/lib")

#dirs paths
include_directories("def")
include_directories("include")
include_directories("$ENV{HOME}/workspace_development/include")
include_directories("$ENV{HOME}/intelFPGA_pro/18.1/hld/host/include")
include_directories("$ENV{HOME}/intelFPGA_pro/18.1/hld/examples_aoc/common/inc")

#files to include
file (GLOB SRC_FILES "src/*.cpp")
file (GLOB DEF_FILES "def/*.h")
file (GLOB INCLUDE_FILES "include/*.h")
# file(GLOB OPENCL_1 "$ENV{HOME}/intelFPGA_pro/18.1/hld/host/include/CL/*.hpp")
# file(GLOB OPENCL_2 "$ENV{HOME}/intelFPGA_pro/18.1/hld/examples_aoc/common/inc/AOCLUtils/*.h")

# #pybind tasks
find_package(pybind11 REQUIRED)
pybind11_add_module(netStandalone
    # ${OPENCL_1}
    # ${OPENCL_2}
    ${DEF_FILES}
    ${SRC_FILES}
    ${INCLUDE_FILES})

#executable
# add_executable(netStandalone
#     ${DEF_FILES}
#     ${SRC_FILES}
#     ${INCLUDE_FILES})

SET(AOCL_COMPILE_FLAGS "-fstack-protector -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fPIE -fPIC")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${AOCL_COMPILE_FLAGS}")

SET(AOCL_LINK_FLAGS "-L/home/hai/intelFPGA_pro/18.1/hld/host/linux64/lib -z noexecstack -Wl,-z,relro,-z,now -Wl,-Bsymbolic -pie -lOpenCL")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${AOCL_LINK_FLAGS}")

set_target_properties(netStandalone PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
target_link_libraries(netStandalone PRIVATE 
    # $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libsvml.so
    # $ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libintlc.so.5
    $ENV{HOME}/intelFPGA_pro/18.1/hld/host/linux64/lib/libOpenCL.so
    ${LIB_FOLDER}/libnetCPU.a
    ${LIB_FOLDER}/libnetFPGA.a)
set_target_properties(netStandalone PROPERTIES INSTALL_RPATH $ENV{HOME}/intelFPGA_pro/18.1/hld/host/linux64/lib) #$ENV{HOME}/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin
set_target_properties(netStandalone PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

#install to conda python lib dir
install(TARGETS netStandalone DESTINATION ${PYTHON_LIBRARY_DIR})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
