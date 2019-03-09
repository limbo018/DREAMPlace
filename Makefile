SITE_PACKAGE = # --user or empty 
CUDAFLAGS = -gencode=arch=compute_60,code=sm_60

.PHONY: dreamplace thirdparty
all: dreamplace 

dreamplace: thirdparty
	make -C dreamplace

thirdparty:
	make -C thirdparty

clean:
	make clean -C thirdparty
	make clean -C dreamplace
