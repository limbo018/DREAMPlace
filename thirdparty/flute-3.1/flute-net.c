#include <stdio.h>
#include <stdlib.h>
#include "flute.h"

int main()
{
    int d=0;
    int x[MAXD], y[MAXD];
    Tree flutetree;
    int flutewl;
    
    while (!feof(stdin)) {
        scanf("%d %d\n", &x[d], &y[d]);
        d++;
    }
    readLUT("POWV9.dat", "POST9.dat");

    flutetree = flute(d, x, y, ACCURACY);
    printf("FLUTE wirelength = %d\n", flutetree.length);

    flutewl = flute_wl(d, x, y, ACCURACY);
    printf("FLUTE wirelength (without RSMT construction) = %d\n", flutewl);
}
