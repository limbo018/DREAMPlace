SITE_PACKAGE = # --user or empty 
PROJECT_ROOT_DIR = $(realpath ../../../)

all:
	@echo "site-package: $(SITE_PACKAGE)"
	cd src; \
		BOOST_DIR=$(BOOST_DIR) \
		LIMBO_DIR=$(PROJECT_ROOT_DIR)/lib/Limbo \
		UTILITY_DIR=$(PROJECT_ROOT_DIR)/lib \
		OPS_DIR=$(PROJECT_ROOT_DIR)/dreamplace/ops \
		python setup.py install $(SITE_PACKAGE); \
		cd ..

clean:
	cd src; python setup.py develop --uninstall; cd ..
	rm -rf src/build/ src/dist src/draw_place.egg-info/
	rm -f *.pyc
	rm -f src/*.pyc
