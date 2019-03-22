#include  <stdlib.h>
#include  <stdio.h>
#include   <assert.h>
#include  "global.h"
#include  "neighbors.h"
#include  "dist.h"
#include  "heap.h"
#include  "err.h"



void  mst2_package_init( long  n )
{
  allocate_heap( n );
  allocate_nn_arrays( n );
}

/****************************************************************************/
/*
*/

void  mst2_package_done()
{
  deallocate_heap();
  deallocate_nn_arrays();
}  

/****************************************************************************/
/*
*/

void  mst2
( 
  long    n,
  Point*  pt, 
  long*   parent
)
{
  long  i, k, nn1;
  long  d;
  long  oct;
  long  root = 0;
  extern  nn_array*  nn;

//  brute_force_nearest_neighbors( n, pt, nn );
  dq_nearest_neighbors( n, pt, nn );

  /* 
     Binary heap implementation of Prim's algorithm.
     Runs in O(n*log(n)) time since at most 8n edges are considered
  */

  heap_init( n );
  heap_insert( root, 0 );
  parent[root] = root;

  for( k = 0;  k < n;  k++ )   /* n points to be extracted from heap */
  {
    i = heap_delete_min();

    if (i<0) break;
#ifdef DEBUG
    assert( i >= 0 );
#endif 

    /*
      pt[i] entered the tree, update heap keys for its neighbors
    */
    for( oct = 0;  oct < 8;  oct++ )
    {
      nn1 = nn[i][oct]; 
      if( nn1 >= 0 )
      {
        d  = dist( pt[i], pt[nn1] );
        if( in_heap(nn1) && (d < heap_key(nn1)) )
        {
          heap_decrease_key( nn1, d );
          parent[nn1] = i;
        } 
        else if( never_seen(nn1) )
        {
          heap_insert( nn1, d );
          parent[nn1] = i;
        }
      }
    }
  }
}

/****************************************************************************/
/****************************************************************************/

