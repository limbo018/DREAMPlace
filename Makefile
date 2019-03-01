SITE_PACKAGE = # --user or empty 

.PHONY: dreamplace thirdparty
all: dreamplace 

dreamplace: thirdparty
	make -C dreamplace

thirdparty:
	make -C thirdparty

clean:
	make clean -C thirdparty
	make clean -C dreamplace
