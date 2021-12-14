cmake_minimum_required(VERSION 3.22.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(vitiate VERSION 0.1.0)

#dirs paths
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/def")
include_directories("${CMAKE_SOURCE_DIR}/src")

#pybind/python dirs
set(pybind11_DIR "${HOME}/anaconda3/envs/Vitiate/share/cmake/pybind11")
set(PYTHON_LIBRARY_DIR "${HOME}anaconda3/envs/Vitiate/lib/python3.10/site-packages")
set(PYTHON_EXECUTABLE "${HOME}/anaconda3/envs/Vitiate/bin/python")

#files to include
file (GLOB SRC_FILES "src/*.cpp")
file (GLOB DEF_FILES "def/*.h")
file (GLOB HEADER_FILES "include/*.h")

# #pybind tasks
# find_package(pybind11 REQUIRED)
# pybind11_add_module(vitiate 
#     ${DEF_FILES}
#     ${HEADER_FILES}
#     ${SRC_FILES})
add_executable(vitiate ${DEF_FILES} ${HEADER_FILES} ${SRC_FILES})

# #install to conda python lib dir
# install(TARGETS vitiate DESTINATION ${PYTHON_LIBRARY_DIR})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
