CC := g++

FMM_DIR := fast_matrix_market
EIGEN_INCLUDE_DIR := /usr/include/eigen3

INCLUDES := $(FMM_DIR)/include \
			$(EIGEN_INCLUDE_DIR)

OPT := -O2
IFLAGS := $(addprefix -I, $(INCLUDES))
CFLAGS := $(IFLAGS) $(OPT)

all:
	$(CC) $(CFLAGS) -o main main.cpp

clean:
	rm -f main
