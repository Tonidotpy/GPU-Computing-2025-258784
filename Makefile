LIB_DIR := lib
LIB_BUILD_DIR := $(LIB_DIR)/build

PROJECTS := $(shell find . -mindepth 1 -maxdepth 1 -type d)
PROJECTS := $(filter-out ./.git ./data ./.cache ./$(LIB_DIR), $(PROJECTS))

LIBS := $(shell find $(LIB_DIR) -mindepth 1 -maxdepth 1 -type d)
LIBS := $(filter-out $(LIB_DIR)/build, $(LIBS))

ifdef DEBUG
OPT := -Og -g2 -ggdb2
else
OPT := -O2
endif

CFLAGS := $(OPT) -Wall -Wextra -Wpedantic -Werror

ifdef NLOGGER
NLOGGER := 1
CFLAGS += -DNLOGGER
endif

all: $(PROJECTS)
	@echo 'Done'

.PHONY: $(PROJECTS)
$(PROJECTS): %: $(LIBS)
	@echo "Building $@ project..."
	cd $@ && make DEBUG=$(DEBUG) LIB_DIR=$(abspath $(LIB_DIR)) NLOGGER=$(NLOGGER)

.PHONY: $(LIBS)
$(LIBS): %: $(LIB_BUILD_DIR)
	$(eval INC := $(addprefix -I$(CURDIR)/, $(shell find $@/include -type d)))
	$(eval SRC := $(addprefix $(CURDIR)/, $(wildcard $@/src/*.c)))
	cd $(LIB_BUILD_DIR) && $(CC) $(CFLAGS) $(INC) -c $(SRC)

$(LIB_BUILD_DIR):
	mkdir -p $@

define \n


endef

.PHONY: clean
clean:
	rm -fr $(LIB_BUILD_DIR)
	$(foreach project, $(PROJECTS), cd $(project) && make clean${\n})
