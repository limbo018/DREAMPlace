SITE_PACKAGE = # --user or empty 
CUDAFLAGS = -gencode=arch=compute_60,code=sm_60

all:
	@echo "site-package: $(SITE_PACKAGE)"
	cd src; CUDAFLAGS=$(CUDAFLAGS) python setup.py install $(SITE_PACKAGE); cd ..

clean:
	cd src; python setup.py develop --uninstall; cd ..
	rm -rf src/build/ src/dist src/electric_potential.egg-info/
	rm -f *.pyc
	rm -f src/*.pyc
