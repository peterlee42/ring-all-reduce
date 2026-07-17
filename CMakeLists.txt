cmake_minimum_required(VERSION 3.16)

project(ring_allreduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

option(ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

include(FetchContent)

FetchContent_Declare(
    googletest
    URL
        https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

# ------------------------------------------------------------------------------
# Training executable
# ------------------------------------------------------------------------------

add_executable(train
    src/train_main.cpp
    src/ml/dataset.cpp
    src/ml/shallow_nn.cpp
    src/distributed/protocol.cpp
    src/distributed/serialization.cpp
)

target_include_directories(train PRIVATE include)

target_compile_options(train PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

# ------------------------------------------------------------------------------
# Serialization tests
# ------------------------------------------------------------------------------

add_executable(test_serialization
    tests/distributed/test_serialization.cpp
    src/distributed/serialization.cpp
)

target_include_directories(test_serialization PRIVATE include)

target_compile_options(test_serialization PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(test_serialization PRIVATE
    GTest::gtest_main
)

# ------------------------------------------------------------------------------
# Protocol tests
# ------------------------------------------------------------------------------

add_executable(test_protocol
    tests/distributed/test_protocol.cpp
    src/distributed/protocol.cpp
    src/distributed/serialization.cpp
)

target_include_directories(test_protocol PRIVATE include)

target_compile_options(test_protocol PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(test_protocol PRIVATE
    GTest::gtest_main
)

# ------------------------------------------------------------------------------
# Sanitizers
# ------------------------------------------------------------------------------

if(ENABLE_SANITIZERS)
    foreach(target
        train
        test_serialization
        test_protocol
    )
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )

        target_link_options(${target} PRIVATE
            -fsanitize=address,undefined
        )
    endforeach()
endif()

# ------------------------------------------------------------------------------
# GoogleTest discovery
# ------------------------------------------------------------------------------

include(GoogleTest)

gtest_discover_tests(test_serialization)
gtest_discover_tests(test_protocol)
