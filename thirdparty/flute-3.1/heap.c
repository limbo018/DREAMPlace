/****************************************************************************/
/*
  Binary heap routines for use in Prim's algorithm, 
  with points are numbered from 0 to n-1
*/

#include <stdlib.h>
#include "heap.h"
#include "err.h"


Heap*   _heap = (Heap*)NULL;
long    _max_heap_size = 0;
long    _heap_size = 0;

/****************************************************************************/
/*
*/

void  allocate_heap( long n )
{
  if( _max_heap_size < n ) 
  {
    _heap = (Heap*)realloc( (void*)_heap, (size_t)(n+1)*sizeof(Heap) ); 
    if( ! _heap )
    {
      err_exit( "Cannot reallocate memory in allocate_heap!" );
    } 
    _max_heap_size = n;
  }
}
/****************************************************************************/
/*
*/

void  deallocate_heap()
{
  _max_heap_size = 0; 
  if( _heap )
  {
    free( (void*)_heap );
    _heap = (Heap*)NULL;
  }
}

/****************************************************************************/

void  heap_init( long  n )
{
  register long  p;

  allocate_heap( n );
  _heap_size = 0;
  for( p = 0;  p < n;  p++ )
  { 
    heap_idx( p ) = 0;
  }
 
} /* END heap_init() */

/****************************************************************************/

void  heap_insert( 
  long   p, 
  long   key 
)
{
  register long  k;       /* hole in the heap     */   
  register long  j;       /* parent of the hole   */
  register long  q;       /* heap_elt(j)          */

  heap_key( p ) = key;

  if( _heap_size == 0 )
  {
    _heap_size = 1;
    heap_elt( 1 ) = p;
    heap_idx( p ) = 1;          
    return;
  }

  k = ++ _heap_size;
  j = k >> 1;            /* k/2 */

  while( (j > 0) && (heap_key(q=heap_elt(j)) > key) ) { 

    heap_elt( k ) = q;
    heap_idx( q ) = k;
    k = j;
    j = k>>1;    /* k/2 */

  }
 
  /* store p in the position of the hole */
  heap_elt( k ) = p;
  heap_idx( p ) = k;      

} /* END heap_insert() */


/****************************************************************************/

void  heap_decrease_key
( 
  long   p, 
  long   new_key 
)
{
  register long    k;       /* hole in the heap     */   
  register long    j;       /* parent of the hole   */
  register long    q;       /* heap_elt(j)          */

  heap_key( p ) = new_key;
  k = heap_idx( p ); 
  j = k >> 1;            /* k/2 */

  if( (j > 0) && (heap_key(q=heap_elt(j)) > new_key) ) { /* change is needed */
    do {

      heap_elt( k ) = q;
      heap_idx( q ) = k;
      k = j;
      j = k>>1;    /* k/2 */

    } while( (j > 0) && (heap_key(q=heap_elt(j)) > new_key) );

    /* store p in the position of the hole */
    heap_elt( k ) = p;
    heap_idx( p ) = k;      
  }

} /* END heap_decrease_key() */


/****************************************************************************/

long  heap_delete_min()
{
  long    min, last;  
  register long  k;         /* hole in the heap     */   
  register long  j;         /* child of the hole    */
  register long  l_key;     /* key of last point    */

  if( _heap_size == 0 )            /* heap is empty */
    return( -1 );

  min  = heap_elt( 1 );
  last = heap_elt( _heap_size -- );
  l_key = heap_key( last );

  k = 1;  j = 2;
  while( j <= _heap_size ) {

    if( heap_key(heap_elt(j)) > heap_key(heap_elt(j+1)) ) 
      j++;

    if( heap_key(heap_elt(j)) >= l_key)  
      break;                     /* found a position to insert 'last' */

    /* else, sift hole down */ 
    heap_elt(k) = heap_elt(j);    /* Note that j <= _heap_size */
    heap_idx( heap_elt(k) ) = k;
    k = j;
    j = k << 1;
  }

  heap_elt( k ) = last;
  heap_idx( last ) = k;

  heap_idx( min ) = -1;   /* mark the point visited */
  return( min );

} /* END heap_delete_min() */


/****************************************************************************/

