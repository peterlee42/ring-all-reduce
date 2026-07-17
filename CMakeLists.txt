cmake_minimum_required(VERSION 3.16)

project(ring_allreduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

option(ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

if(ENABLE_SANITIZERS)
    target_compile_options(train PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    target_link_options(train PRIVATE
        -fsanitize=address,undefined
    )
endif()
