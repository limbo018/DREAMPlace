#include "global.h"

void  allocate_nn_arrays( long n );
void  deallocate_nn_arrays();

void  brute_force_nearest_neighbors
(
  long       n,
  Point*     pt,
  nn_array*  nn
);

void  dq_nearest_neighbors
(
  long       n,
  Point*     pt,
  nn_array*  nn
);

