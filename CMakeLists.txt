cmake_minimum_required(VERSION 3.0.0)
project(Vitiate VERSION 0.1.0)

include(CTest)
enable_testing()

INCLUDE_DIRECTORIES(./ ./utils/classes ./utils/python)
# ADD_SUBDIRECTORY(utils/classes)


add_subdirectory(utils/python)
add_subdirectory(utils/pybind11)
add_executable(Main main.cpp)

target_link_libraries(Main PRIVATE pybind11::embed)
target_link_libraries (Main LINK_PUBLIC mypy)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
