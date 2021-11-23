cmake_minimum_required(VERSION 3.0.0)
project(Vitiate VERSION 0.1.0)

include(CTest)
enable_testing()

INCLUDE_DIRECTORIES(./ ./utils/classes)

add_subdirectory(utils/pybind11)
add_executable(main main.cpp)

target_link_libraries(main PRIVATE pybind11::embed)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
