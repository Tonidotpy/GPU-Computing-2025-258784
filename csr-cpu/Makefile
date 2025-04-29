TARGET := SpMV

BUILD_DIR := build
SRC_DIR := src
INCLUDE_DIR := include
LIB_DIR ?= ../lib
LIB_BUILD_DIR := $(LIB_DIR)/cbuild
LIB_INCLUDES := $(shell find $(LIB_DIR) -name include -type d)

SOURCES := $(wildcard $(SRC_DIR)/*.c)
INCLUDES := $(shell find $(INCLUDE_DIR) -type d)
INCLUDES += $(LIB_INCLUDES)
OBJECTS := $(SOURCES:%.c=$(BUILD_DIR)/%.o)
OBJECTS += $(wildcard $(LIB_BUILD_DIR)/*.o)

ifeq ($(DEBUG),1)
OPT := -Og -g2 -ggdb2
else
OPT := -O2
endif

IFLAGS := $(addprefix -I, $(INCLUDES))
CFLAGS := $(IFLAGS) $(OPT) -Wall -Wextra -Wpedantic -Werror
LDFLAGS := -lm

ifeq ($(NLOGGER),1)
CFLAGS += -DNLOGGER
endif

$(BUILD_DIR)/$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
