cmake_minimum_required(VERSION 3.16)
project(ring_allreduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)
option(ENABLE_TCP_TESTS "Also run integration tests over localhost TCP" OFF)
include(CTest)
find_package(Threads REQUIRED)

if(NOT UNIX)
    message(FATAL_ERROR "The TCP transport requires POSIX sockets (Linux or macOS)")
endif()

function(project_options target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(ENABLE_SANITIZERS)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

add_library(ring_core
    src/distributed/protocol.cpp
    src/distributed/serialization.cpp
    src/distributed/ring.cpp
    src/ml/dataset.cpp
    src/ml/shallow_nn.cpp
)
target_include_directories(ring_core PUBLIC include)
target_link_libraries(ring_core PUBLIC Threads::Threads)
project_options(ring_core)

add_executable(train src/train_main.cpp)
target_link_libraries(train PRIVATE ring_core)
project_options(train)

if(BUILD_TESTING)
    foreach(name serialization protocol ml)
        if(name STREQUAL "ml")
            set(test_source tests/ml/test_ml.cpp)
        else()
            set(test_source tests/distributed/test_${name}.cpp)
        endif()
        add_executable(test_${name} ${test_source} tests/test_main.cpp)
        target_include_directories(test_${name} PRIVATE tests)
        target_link_libraries(test_${name} PRIVATE ring_core)
        project_options(test_${name})
        add_test(NAME ${name} COMMAND test_${name})
    endforeach()
    add_executable(test_ring_worker tests/distributed/test_ring_worker.cpp)
    target_link_libraries(test_ring_worker PRIVATE ring_core)
    project_options(test_ring_worker)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
    add_test(NAME integration
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/test_ring.py
            --train $<TARGET_FILE:train> --worker $<TARGET_FILE:test_ring_worker>)
    set_tests_properties(integration PROPERTIES TIMEOUT 120)
    if(ENABLE_TCP_TESTS)
        add_test(NAME integration_tcp
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/test_ring.py
                --train $<TARGET_FILE:train> --worker $<TARGET_FILE:test_ring_worker> --transport tcp)
        set_tests_properties(integration_tcp PROPERTIES TIMEOUT 120)
    endif()
endif()
