SITE_PACKAGE = # --user or empty 

.PHONY: utility dct hpwl rmst_wl logsumexp_wirelength weighted_average_wirelength density_overflow density_potential electric_potential place_io move_boundary greedy_legalize draw_place
all: utility dct hpwl rmst_wl logsumexp_wirelength weighted_average_wirelength density_overflow density_potential electric_potential place_io move_boundary greedy_legalize draw_place 

utility:
	make -C utility 

dct: utility
	make -C dct

hpwl: utility
	make -C hpwl

rmst_wl: utility 
	make -C rmst_wl

logsumexp_wirelength: utility
	make -C logsumexp_wirelength

weighted_average_wirelength: utility 
	make -C weighted_average_wirelength

density_overflow: utility 
	make -C density_overflow

density_potential: utility 
	make -C density_potential

electric_potential: dct utility 
	make -C electric_potential

place_io: utility 
	make -C place_io

move_boundary: utility 
	make -C move_boundary

greedy_legalize: utility 
	make -C greedy_legalize

draw_place: utility 
	make -C draw_place

clean:
	make clean -C utility 
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
