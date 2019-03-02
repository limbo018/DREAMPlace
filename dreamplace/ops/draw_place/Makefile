SITE_PACKAGE = # --user or empty 
PROJECT_ROOT_DIR = $(realpath ../../../)

ENV_CONFIG = \
		BOOST_DIR=$(BOOST_DIR) \
		LIMBO_DIR=$(PROJECT_ROOT_DIR)/lib/Limbo \
		UTILITY_DIR=$(PROJECT_ROOT_DIR)/lib \
		OPS_DIR=$(PROJECT_ROOT_DIR)/dreamplace/ops 

all:
	@echo "site-package: $(SITE_PACKAGE)"
	cd src; $(ENV_CONFIG) python setup.py install $(SITE_PACKAGE); cd ..

clean:
	cd src; $(ENV_CONFIG) python setup.py develop --uninstall; cd ..
	rm -rf src/build/ src/dist src/draw_place.egg-info/
	rm -f *.pyc
	rm -f src/*.pyc
