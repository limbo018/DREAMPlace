PROJECT_ROOT_DIR = $(realpath ../../../)
SRCS_DIR = $(realpath src)
OBJ_DIR = $(PROJECT_ROOT_DIR)/obj/dreamplace/ops/utility
LIB_DIR = $(PROJECT_ROOT_DIR)/lib
TARGET = $(LIB_DIR)/libutility.a
INCLUDE = -I $(PROJECT_ROOT_DIR)/dreamplace/ops

include $(PROJECT_ROOT_DIR)/Include.mk

all: $(TARGET)

$(OBJ_DIR)/Msg.o: src/Msg.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE)

$(TARGET): $(OBJ_DIR)/Msg.o
	@mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) -o $@

clean:
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR)
	rm -rf src/build/ src/dist src/move_boundary.egg-info/
	rm -f *.pyc
	rm -f src/*.pyc
