# Convenience wrapper: keep generated CMake files in build/.
CMAKE ?= cmake
BUILD_DIR ?= build
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=Release
JOBS ?= 4

.PHONY: all configure test demo train
all: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

demo: all
	python3 scripts/run_local.py --binary $(BUILD_DIR)/train

train: all
	python3 scripts/run_local.py --binary $(BUILD_DIR)/train --workers 4 -- data/toy.csv --input 2 --hidden 8 --classes 2 --rounds 100 --lr 0.1 --threshold 0
