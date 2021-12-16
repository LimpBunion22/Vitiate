cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(vitiate VERSION 0.1.0)

#dirs paths
include_directories("include")
include_directories("def")
include_directories("src")
include_directories("cpu")

#pybind/python dirs
set(pybind11_DIR "$ENV{HOME}/anaconda3/envs/Vitiate/share/cmake/pybind11")
set(PYTHON_LIBRARY_DIR "$ENV{HOME}/anaconda3/envs/Vitiate/lib/python3.8/site-packages")
set(PYTHON_EXECUTABLE "$ENV{HOME}/anaconda3/envs/Vitiate/bin/python")

#files to include
file (GLOB SRC_FILES "src/*.cpp")
file (GLOB DEF_FILES "def/*.h")
file (GLOB HEADER_FILES "include/*.h")
file (GLOB CPU_HEADERS "cpu/*.h")
file (GLOB CPU_SOURCES "cpu/*.cpp")

#pybind tasks
find_package(pybind11 REQUIRED)
pybind11_add_module(vitiate 
    ${DEF_FILES}
    ${HEADER_FILES}
    ${SRC_FILES}
    ${CPU_HEADERS}
    ${CPU_SOURCES})
# add_library(vitiate MODULE
#     ${DEF_FILES}
#     ${HEADER_FILES}
#     ${SRC_FILES}
#     ${CPU_HEADERS}
#     ${CPU_SOURCES})

# target_link_libraries(vitiate PRIVATE pybind11::module pybind11::lto)

# pybind11_extension(vitiate)
# pybind11_strip(vitiate)


# #execute 
# add_executable(vitiate 
#     ${DEF_FILES}
#     ${HEADER_FILES}
#     ${SRC_FILES}
#     ${CPU_HEADERS}
#     ${CPU_SOURCES})

#install to conda python lib dir
install(TARGETS vitiate DESTINATION ${PYTHON_LIBRARY_DIR})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
