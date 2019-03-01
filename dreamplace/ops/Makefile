SITE_PACKAGE = # --user or empty 

all:
	make -C dct
	make -C hpwl
	make -C rmst_wl
	make -C logsumexp_wirelength
	make -C weighted_average_wirelength
	make -C density_overflow
	make -C density_potential
	make -C electric_potential
	make -C place_io
	make -C move_boundary
	make -C greedy_legalize
	make -C draw_place

clean:
	make clean -C dct
	make clean -C hpwl
	make clean -C rmst_wl
	make clean -C logsumexp_wirelength
	make clean -C weighted_average_wirelength
	make clean -C density_overflow
	make clean -C density_potential
	make clean -C electric_potential
	make clean -C place_io
	make clean -C move_boundary
	make clean -C greedy_legalize
	make clean -C draw_place
