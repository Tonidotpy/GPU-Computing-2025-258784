CC := gcc
NVCC := nvcc

LIB_DIR := lib
CLIB_BUILD_DIR := $(LIB_DIR)/cbuild
CULIB_BUILD_DIR := $(LIB_DIR)/cubuild

PROJECTS := $(shell find . -mindepth 1 -maxdepth 1 -type d -name 'csr-*')

LIBS := $(shell find $(LIB_DIR) -mindepth 1 -maxdepth 1 -type d)
LIBS := $(filter-out $(CLIB_BUILD_DIR) $(CULIB_BUILD_DIR), $(LIBS))

ifdef DEBUG
COPT := -Og -g2 -ggdb2
CUOPT := -g -G
else
COPT := -O2
CUOPT := -O2
endif

CFLAGS := $(COPT) -Wall -Wextra -Wpedantic -Werror
CUFLAGS := $(CUOPT) -Werror=all-warnings

ifdef NLOGGER
NLOGGER := 1
CFLAGS += -DNLOGGER
CUFLAGS += -DNLOGGER
endif

all: $(PROJECTS)
	@echo 'Done'

.PHONY: $(PROJECTS)
$(PROJECTS): %: $(LIBS)
	@echo "Building $@ project..."
	cd $@ && make DEBUG=$(DEBUG) LIB_DIR=$(abspath $(LIB_DIR)) NLOGGER=$(NLOGGER)

.PHONY: $(LIBS)
$(LIBS): %: $(CLIB_BUILD_DIR) $(CULIB_BUILD_DIR)
	$(eval INC := $(addprefix -I$(CURDIR)/, $(shell find $@/include -type d)))
	$(eval CSRC := $(addprefix $(CURDIR)/, $(wildcard $@/src/*.c)))
	$(eval CUSRC := $(addprefix $(CURDIR)/, $(wildcard $@/src/*.cu)))
	$(if $(CSRC), cd $(CLIB_BUILD_DIR) && $(CC) $(CFLAGS) $(INC) -c $(CSRC),)
	$(if $(CUSRC), cd $(CULIB_BUILD_DIR) && $(NVCC) $(CUFLAGS) $(INC) -c $(CUSRC),)

$(CLIB_BUILD_DIR):
	mkdir -p $@

$(CULIB_BUILD_DIR):
	mkdir -p $@

define \n


endef

.PHONY: clean
clean:
	rm -fr $(CLIB_BUILD_DIR)
	rm -fr $(CULIB_BUILD_DIR)
	$(foreach project, $(PROJECTS), cd $(project) && make clean${\n})
