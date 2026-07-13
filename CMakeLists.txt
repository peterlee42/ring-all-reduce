cmake_minimum_required(VERSION 3.16)

project(ring_allreduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(train_local
    src/train_main.cpp
    src/dataset.cppa
    src/shallow_nn.cpp
)

target_include_directories(train_local PRIVATE include)

target_compile_options(train_local PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

option(ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

if(ENABLE_SANITIZERS)
    target_compile_options(train_local PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    target_link_options(train_local PRIVATE
        -fsanitize=address,undefined
    )
endif()
