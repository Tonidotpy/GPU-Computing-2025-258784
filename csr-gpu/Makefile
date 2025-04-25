CC := gcc
NVCC := nvcc

TARGET := SpMV

BUILD_DIR := build
SRC_DIR := src
INCLUDE_DIR := include
LIB_DIR ?= ../lib
LIB_BUILD_DIR := $(LIB_DIR)/build
LIB_INCLUDES := $(shell find $(LIB_DIR) -name include -type d)

C_SOURCES := $(wildcard $(SRC_DIR)/*.c)
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
INCLUDES := $(shell find $(INCLUDE_DIR) -type d)
INCLUDES += $(LIB_INCLUDES)
C_OBJECTS := $(C_SOURCES:%.c=$(BUILD_DIR)/%.o)
C_OBJECTS += $(wildcard $(LIB_BUILD_DIR)/*.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(BUILD_DIR)/%.o)

ifeq ($(DEBUG),1)
COPT := -Og -g2 -ggdb2
CUOPT := -g -G
else
COPT := -O2
CUOPT := -O2
endif

CUOPT += -diag-suppress 550

IFLAGS := $(addprefix -I, $(INCLUDES))
CFLAGS := $(IFLAGS) $(COPT) -Wall -Wextra -Wpedantic -Werror
CUFLAGS := $(IFLAGS) $(CUOPT) -Werror=all-warnings
LDFLAGS := -lm

ifeq ($(NLOGGER),1)
CFLAGS += -DNLOGGER
CUFLAGS += -DNLOGGER
endif

$(BUILD_DIR)/$(TARGET): $(C_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(C_OBJECTS) $(CU_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
