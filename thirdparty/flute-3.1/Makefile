#CC = gcc
CFLAGS = -O3 -fPIC -I.

SRC     = dist.c dl.c err.c heap.c mst2.c neighbors.c \
	bookshelf_IO.c memAlloc.c flute.c flute_mst.c
OBJ     = $(SRC:.c=.o)
LIB     = libflute.a

all: flute-net flute-ckt rand-pts $(LIB)

$(LIB): dist.o dl.o err.o heap.o mst2.o neighbors.o \
	bookshelf_IO.o memAlloc.o flute.o flute_mst.o
	ar rvs $@ $^

flute-net: flute-net.c ${OBJ}
	$(CC) $(CFLAGS) -o flute-net flute-net.c ${OBJ} -lm

flute-ckt: flute-ckt.c ${OBJ}
	$(CC) $(CFLAGS) -o flute-ckt flute-ckt.c ${OBJ} -lm

rand-pts: rand-pts.c
	$(CC) $(CFLAGS) -o rand-pts rand-pts.c 

flute.o: flute.c flute.h
	$(CC) $(CFLAGS) -c -o flute.o flute.c

flute_mst.o: flute_mst.c flute.h
	$(CC) $(CFLAGS) -c -o flute_mst.o flute_mst.c

install:
	@mkdir -p ${PREFIX}/include
	cp flute.h ${PREFIX}/include
	@mkdir -p ${PREFIX}/lib
	cp libflute.a ${PREFIX}/lib

clean:
	rm -f *.o core*
