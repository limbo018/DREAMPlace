PROJECT_ROOT_DIR = $(realpath ../../../)
OP_NAME = place_io
LIB_DIR = $(PROJECT_ROOT_DIR)/lib/dreamplace/ops/$(OP_NAME)
BUILD_DIR = $(PROJECT_ROOT_DIR)/build/dreamplace/ops/$(OP_NAME)

ENV_CONFIG = \
		BOOST_DIR=$(BOOST_DIR) \
		LIMBO_DIR=$(PROJECT_ROOT_DIR)/lib/Limbo \
		UTILITY_DIR=$(PROJECT_ROOT_DIR)/lib \
		OPS_DIR=$(PROJECT_ROOT_DIR)/dreamplace/ops 

all:
	$(ENV_CONFIG) python setup.py build --build-temp=$(BUILD_DIR) --build-lib=$(LIB_DIR)

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)
	rm -f *.pyc
	rm -f src/*.pyc
