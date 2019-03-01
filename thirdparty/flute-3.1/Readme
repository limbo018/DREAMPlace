-------------- FLUTE - Version 3.1 -----------------
                       by
                Chris C.-N. Chu
       Dept. of ECpE, Iowa State University
             Copyright (c) - 2005
  Iowa State University Research Foundation, Inc.
----------------------------------------------------

This package contains the following files:

 flute.c -- The rectilinear Steiner minimal tree and wirelength estimation
	    algorithm described in the ICCAD 04 and ISPD 05 papers with
	    some improvements described in TCAD 07 paper.
 flute.h -- The interface to use flute.
 flute_mst.c -- The net breaking and merging techniques described in the
	    VLSIDAT 08 paper.
 dist.[ch], dl.[ch], err.[ch], heap.[ch], mst2.[ch], neighbors.[ch],
	    global.h -- Utility functions used by flute_mst.c
 POWV9.dat -- The lookup-table of optimal POWVs up to degree 9.
 POST9.dat -- The lookup-table for optimal Steiner trees up to degree 9.
 flute-net.c -- A program to evaluate the wirelength of a net. It takes
	    input from stdin as a list of points.
 rand-pts.c -- A program to generate a list of random points.
 flute-ckt.c -- A program to find FLUTE and half-perimeter wirelength
	    of a circuit in bookshelf format.
 bookshelf_IO.[ch] -- Functions for flute-ckt.c to read bookshelf files.
 memAlloc.[ch] -- Functions for flute-ckt.c to allocate memory.
 ibm01/ibm01.* -- ibm01 bookshelf files that can be read by flute-ckt.c
 license.txt -- License agreement.
 ChangeLog.txt
 Makefile
 Readme

To run the programs, first do a 'make'. POWV9.dat and POST9.dat is assume
to be in the current directory.  Some example commands:

 rand-pts | flute-net
 rand-pts 20 | flute-net	    // 20-pin nets
 rand-pts -r 20 | flute-net	    // randomized
 flute-ckt ibm01 ibm01.aux ibm01/ibm01.pl
