SITE_PACKAGE = # --user or empty 
PROJECT_ROOT_DIR = $(realpath ../../../)

ENV_CONFIG = \
		OPS_DIR=$(PROJECT_ROOT_DIR)/dreamplace/ops \
		UTILITY_DIR=$(PROJECT_ROOT_DIR)/lib \
		FLUTE_DIR=$(PROJECT_ROOT_DIR)/lib/flute

all:
	@echo "site-package: $(SITE_PACKAGE)"
	cd src; $(ENV_CONFIG) python setup.py install $(SITE_PACKAGE); cd ..

clean:
	cd src; $(ENV_CONFIG) python setup.py develop --uninstall; cd ..
	rm -rf src/build/ src/dist src/rmst_wl.egg-info/
	rm -f *.pyc
	rm -f src/*.pyc
