PROJECT_ROOT_DIR = $(realpath ../../../)
OP_NAME = rmst_wl
LIB_DIR = $(PROJECT_ROOT_DIR)/lib/dreamplace/ops/$(OP_NAME)
BUILD_DIR = $(PROJECT_ROOT_DIR)/build/dreamplace/ops/$(OP_NAME)

ENV_CONFIG = \
		OPS_DIR=$(PROJECT_ROOT_DIR)/dreamplace/ops \
		UTILITY_DIR=$(PROJECT_ROOT_DIR)/lib \
		FLUTE_DIR=$(PROJECT_ROOT_DIR)/lib/flute

all:
	$(ENV_CONFIG) python setup.py build --build-temp=$(BUILD_DIR) --build-lib=$(LIB_DIR)

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)
	rm -f *.pyc
	rm -f src/*.pyc
