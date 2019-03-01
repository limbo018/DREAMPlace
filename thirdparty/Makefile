PROJECT_ROOT_DIR = $(realpath ..)

.PHONY: flute limbo 
all: flute limbo 

.PHONY: clean_flute clean_limbo 
clean: clean_flute clean_limbo 

include $(PROJECT_ROOT_DIR)/Include.mk 

# ============= Install flute ============

FLUTE_DIR = $(PROJECT_ROOT_DIR)/lib/flute

flute: 
	@echo ">> build flute to $(FLUTE_DIR)"
	mkdir -p $(FLUTE_DIR)
	make -C flute-3.1
	make install -C flute-3.1 PREFIX=$(FLUTE_DIR)

clean_flute:
	@echo ">> clean flute"
	make clean -C flute-3.1

# =============== End flute ===============

# ============= Install Limbo ============

LIMBO_DIR = $(PROJECT_ROOT_DIR)/lib/Limbo

limbo: 
	@echo ">> build Limbo to $(LIMBO_DIR)"
	mkdir -p $(LIMBO_DIR)
	make CXX=$(CXX) CC=$(CC) FC=$(FC) CXXSTD="-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0" -C Limbo
	make install -C Limbo PREFIX=$(LIMBO_DIR)

clean_limbo:
	@echo ">> clean Limbo"
	make clean -C Limbo

# =============== End Limbo ===============
