PROJECT_ROOT_DIR = $(realpath ../../..)
OPS_DIR = $(PROJECT_ROOT_DIR)/dreamplace/ops
INCLUDE = -I $(OPS_DIR)

all: abacus_unitest 

abacus_unitest: abacus_unitest.cpp $(OPS_DIR)/utility/src/Msg.cpp
	g++ -g -O2 -std=c++11 $^ -o $@ $(INCLUDE)

clean: 
	rm -f abacus_unitest
