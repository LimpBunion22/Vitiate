
add_library(mypy SHARED python_handler.h python_handler.cxx)
target_include_directories (mypy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_sources(Vitiate2 PRIVATE python_handler.cpp)