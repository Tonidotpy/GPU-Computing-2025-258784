CC := g++

TARGET := main

BUILD_DIR := build
FMM_DIR := fast_matrix_market
EIGEN_INCLUDE_DIR := /usr/include/eigen3

INCLUDES := $(FMM_DIR)/include \
			$(EIGEN_INCLUDE_DIR)
SOURCES := main.cpp

OPT := -O2
IFLAGS := $(addprefix -I, $(INCLUDES))
CFLAGS := $(IFLAGS) $(OPT)

$(BUILD_DIR)/$(TARGET): $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $(SOURCES)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -f main
