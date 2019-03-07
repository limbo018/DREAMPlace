SITE_PACKAGE = # --user or empty 
PROJECT_ROOT_DIR = $(realpath ../../../)

CUDAFLAGS = -gencode=arch=compute_60,code=sm_60
ENV_CONFIG = \
		OPS_DIR=$(PROJECT_ROOT_DIR)/dreamplace/ops \
		CUDAFLAGS=$(CUDAFLAGS)

all:
	@echo "site-package: $(SITE_PACKAGE)"
	cd src; $(ENV_CONFIG) python setup.py install $(SITE_PACKAGE); cd ..

clean:
	cd src; $(ENV_CONFIG) python setup.py develop --uninstall; cd ..
	rm -rf src/build/ src/dist src/density_potential.egg-info/
	rm -f *.pyc
	rm -f src/*.pyc
