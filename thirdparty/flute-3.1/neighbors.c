#include  <assert.h>
#include  <string.h>
#include  <stdlib.h>
#include  "global.h"
#include  "err.h"
#include  "dist.h"

long  octant
(
  Point  from,
  Point  to
);

static Point* _pt;

/***************************************************************************/
/*
  For efficiency purposes auxiliary arrays are allocated as globals 
*/

long    max_arrays_size = 0;
nn_array*  nn   = (nn_array*)NULL;
Point*  sheared = (Point*)NULL;
long*  sorted   = (long*)NULL;
long*  aux      = (long*)NULL;  

/***************************************************************************/
/*
  resize the auxiliary arrays to fit the specified number of points 
*/

void  allocate_nn_arrays( long  n )
{
  if( max_arrays_size < n ) 
  {
    nn      = (nn_array*)realloc( (void*)nn, (size_t)n*sizeof(nn_array) );
    sheared = (Point*)realloc( (void*)sheared, (size_t)n*sizeof(Point) );
    sorted  = (long*)realloc( (void*)sorted, (size_t)n*sizeof(long) );
    aux     = (long*)realloc( (void*)aux, (size_t)n*sizeof(long) );
    if( !nn || !sheared || !sorted || !aux )
    {
      err_exit( "Cannot allocate memory in allocate_nn_arrays!" );
    }
    max_arrays_size = n;
  }
}

/***************************************************************************/
/*
  free memory used by auxiliary arrays
*/

void  deallocate_nn_arrays()
{
  max_arrays_size = 0;
  if( nn )
  {
    free( (void*)nn );
    nn = (nn_array*)NULL;
  }
  if( sheared )
  {
    free( (void*)sheared );
    sheared = (Point*)NULL;
  }
  if( sorted )
  {
    free( (void*)sorted );
    sorted = (long*)NULL;
  }
  if( aux )
  {
    free( (void*)aux );
    aux = (long*)NULL;
  }

}

/***************************************************************************/
/*
  comparison function for use in quicksort
*/

static  int compare_x
( 
  const void*  i, 
  const void*  j 
)
{
  /*
    points with the same x must appear in increasing order of y 
  */
  if( sheared[*((long*)i)].x == sheared[*((long*)j)].x)
  {
    return  sheared[*((long*)i)].y - sheared[*((long*)j)].y;
  }
  else
  {
    return  sheared[*((long*)i)].x - sheared[*((long*)j)].x;
  }
}


/***************************************************************************/
/*
  Combine step of the Guibas-Stolfi divide-and-conquer NE nearest neighbor
  algorithm. For efficiency purposes SW nearest neighbors are computed 
  at the same time.
*/

void  ne_sw_combine
(
  long    left,
  long    mid,
  long    right,
  Point*  pt,
  long*   sorted,
  long*   aux,
  long    oct,
  nn_array*  nn
)
{
  long   i, j, k, y2; 
  long   i1;
  long   i2; 
  long   best_i2;     /* index of current best nearest-neighbor */
  long   best_dist;   /* distance to best nearest-neighbor      */
  long   d;

#ifdef DEBUG
  assert( right > mid );
  assert( mid > left );
#endif

  /*
    update north-east nearest neighbors accross the mid-line
  */

  i1 = left;
  i2 = mid;   y2 = pt[ sorted[i2] ].y;

  while( (i1 < mid) && (pt[ sorted[i1] ].y >= y2) )
  {
    i1++;
  }
  
  if( i1 < mid )
  {
    best_i2   = i2;
    best_dist = dist2( pt + sorted[i1], pt + sorted[best_i2] );
    i2++;

    while( (i1 < mid) && (i2 < right) )
    {
      if( pt[ sorted[i1] ].y < pt[ sorted[i2] ].y )
      {
        d = dist2( pt + sorted[i1], pt + sorted[i2] );
        if( d < best_dist ) 
        {
          best_i2   = i2;
          best_dist = d;
        }
        i2++;
      }
      else 
      {
        if( (nn[ sorted[i1] ][oct] == -1) || 
            ( best_dist < dist2( pt + sorted[i1], pt + nn[ sorted[i1] ][oct]) ) 
           )
        {
          nn[ sorted[i1] ][oct] = sorted[best_i2];
        }
        i1++;
        if( i1 < mid )
        {
          best_dist = dist2( pt + sorted[i1], pt + sorted[best_i2] );
        }
      }    
    }

    while( i1 < mid )
    {
      if( (nn[ sorted[i1] ][oct] == -1) || 
          ( dist2( pt + sorted[i1], pt + sorted[best_i2] ) < 
            dist2( pt + sorted[i1], pt + nn[ sorted[i1] ][oct]) ) 
        )
      {
        nn[ sorted[i1] ][oct] = sorted[best_i2];
      }
      i1++;
    }
  }
  /*
    repeat for south-west nearest neighbors
  */

  oct = (oct + 4) % 8;

  i1 = right - 1;
  i2 = mid - 1;   y2 = pt[ sorted[i2] ].y;
     
  while( (i1 >= mid) && (pt[ sorted[i1] ].y <= y2) )
  {
    i1--;
  }

  if( i1 >= mid )
  {
    best_i2   = i2;
    best_dist = dist2( pt + sorted[i1], pt + sorted[best_i2] );
    i2--;

    while( (i1 >= mid) && (i2 >= left) )
    {
      if( pt[ sorted[i1] ].y > pt[ sorted[i2] ].y )
      {
        d = dist2( pt + sorted[i1], pt + sorted[i2] );
        if( d < best_dist ) 
        {
          best_i2   = i2;   
          best_dist = d;
        }
        i2--;
      }
      else 
      {
        if( (nn[ sorted[i1] ][oct] == -1) || 
            ( best_dist < dist2( pt + sorted[i1], pt + nn[ sorted[i1] ][oct]) ) 
           )
        {
          nn[ sorted[i1] ][oct] = sorted[best_i2];
        }
        i1--;
        if( i1 >= mid )
        {
          best_dist = dist2( pt + sorted[i1], pt + sorted[best_i2] );
        }
      }    
    }

    while( i1 >= mid )
    {
      if( (nn[ sorted[i1] ][oct] == -1) || 
          ( dist2( pt + sorted[i1], pt + sorted[best_i2] ) < 
            dist2( pt + sorted[i1], pt + nn[ sorted[i1] ][oct]) ) 
        )
      {
        nn[ sorted[i1] ][oct] = sorted[best_i2];
      }
      i1--;
    }
  }

  /*
    merge sorted[left..mid-1] with sorted[mid..right-1] by y-coordinate
  */

  i = left;  /* first unprocessed element in left  list  */
  j = mid;   /* first unprocessed element in right list  */
  k = left;  /* first free available slot in output list */

  while( (i < mid) && (j < right) )
  {
    if( pt[ sorted[i] ].y >= pt[ sorted[j] ].y )
    {
      aux[k++] = sorted[i++]; 
    }
    else 
    {
      aux[k++] = sorted[j++]; 
    }
  }

  /*
    copy leftovers 
  */
  while( i < mid   ) {  aux[k++] = sorted[i++]; }
  while( j < right ) {  aux[k++] = sorted[j++]; }

  /*
    now copy sorted points from 'aux' to 'sorted' 
  */

  for( i = left;  i < right;  i++ )  { sorted[i] = aux[i]; }

#if 0
  memcpy( (void*)(sorted+left),             /* destination */
          (void*)(aux+left),             /* source      */
          (size_t)(right-left)*sizeof(long) /* number of bytes */ 
        );
#endif

}

/***************************************************************************/
/*
   compute north-east and south-west nearest neighbors for points indexed 
   by {sorted[left],...,sorted[right-1]} 
*/

void  ne_sw_nearest_neighbors
(
  long    left,
  long    right,
  Point*  pt,
  long*   sorted,
  long*   aux,
  long    oct,
  nn_array*  nn
)
{
  long   mid;

#ifdef DEBUG
  assert( right > left );
#endif

  if( right == left + 1 )  
  {
    nn[ sorted[left] ][oct] = nn[ sorted[left]][(oct+4) % 8] = -1;
  }
  else
  {
    mid = (left + right) / 2;
    ne_sw_nearest_neighbors( left, mid, pt, sorted, aux, oct, nn );
    ne_sw_nearest_neighbors( mid, right, pt, sorted, aux, oct, nn );
    ne_sw_combine( left, mid, right, pt, sorted, aux, oct, nn );
  }
}

/***************************************************************************/
/*
  Guibas-Stolfi algorithm for computing nearest NE neighbors
*/

void  dq_nearest_neighbors
(
  long      n,
  Point*    pt,
  nn_array*  nn
)
{
  long   i, oct;
  void  check_nn( long, Point*, nn_array* );

  long   shear[4][4] = {
                         {1, -1,  0,  2}, 
                         {2,  0, -1,  1}, 
                         {1,  1, -2,  0}, 
                         {0,  2, -1, -1} 
                       };



_pt = pt;

  for( oct = 0;  oct < 4;  oct++ )
  {
    for( i = 0;   i < n;   i++ )
    {
      sheared[i].x = shear[oct][0]*pt[i].x + shear[oct][1]*pt[i].y;
      sheared[i].y = shear[oct][2]*pt[i].x + shear[oct][3]*pt[i].y;
      sorted[i] = i;
    }
    
    qsort( sorted, n, sizeof(long), compare_x );
    ne_sw_nearest_neighbors( 0, n, sheared, sorted, aux, oct, nn );
  }

#ifdef DEBUG
  check_nn( n, pt, nn );
#endif

}

/***************************************************************************/
/***************************************************************************/
/*
  Brute-force nearest-neighbor computation for debugging purposes
*/

/***************************************************************************/
/*
  Half-open octants are numbered from 0 to 7 in anti-clockwise order 
  starting from ( dx >= dy > 0 ).
*/

#define sgn(x)  ( x>0 ? 1 : (x < 0 ? -1 : 0) )

long  octant
( 
  Point  from,
  Point  to
)
{
  long  dx = to.x - from.x;
  long  dy = to.y - from.y;
  long  sgn1 = sgn(dx)*sgn(dy);
  long  sgn2 = sgn(dx+dy)*sgn(dx-dy);
  long   oct = 0x0;

  
  if( (dy < 0) || ((dy==0) && (dx>0)) )        oct += 4;
  if( (sgn1 < 0) || (dy==0) )                  oct += 2;
  if( (sgn1*sgn2 < 0) || (dy==0) || (dx==0) )  oct += 1;

  return  oct;
}

/***************************************************************************/
/*
  O(n^2) algorithm for computing all nearest neighbors
*/

void  brute_force_nearest_neighbors
(
  long    n,
  Point*  pt,
  nn_array*  nn
)
{
  long  i, j, oct;
  long  d;

  /*
    compute nearest neighbors by inspecting all pairs of points 
  */
  for( i = 0;   i < n;   i++ )
  {
    for( oct = 0;  oct < 8;  oct++ )
    {
      nn[i][oct]   = -1;
    }
  }

  for( i = 0;   i < n;  i++ )
  {
    for( j = i+1;   j < n;  j++ )
    {
      d = dist(pt[i], pt[j]);

      oct = octant( pt[i], pt[j] ); 
      if( ( nn[i][oct] == -1 ) ||
          ( d < dist(pt[i], pt[ nn[i][oct] ]) )
        )
      {
        nn[i][oct]  = j;
      }

      oct = (oct + 4) % 8;       
      if( ( nn[j][oct] == -1 ) ||
          ( d < dist(pt[j], pt[ nn[j][oct] ]) )
        )
      {
        nn[j][oct]  = i;
      }
    }
  }
}


/***************************************************************************/
/*
  compare nearest neighbors against those computed by brute force
*/

void  check_nn
(
  long    n,
  Point*  pt,
  nn_array*  nn
)
{
  long       i, j, oct;
  nn_array*  nn1;

  nn1  = (nn_array*)calloc( (size_t)n, (size_t)sizeof(nn_array) );
  brute_force_nearest_neighbors( n, pt, nn1 );

  for( i = 0;   i < n;   i++ )
  {
    for( oct = 0;  oct < 8;  oct++ )
    {
      if( nn[i][oct] == -1 )
      {
        assert( nn1[i][oct] == -1 );
      }
      else
      {
        assert( nn1[i][oct] != -1 );

        if( octant(pt[i], pt[ nn[i][oct] ]) != oct )
        {
        printf( "WRONG OCTANT!\noct=%ld\n", oct );
        printf( "i=%ld, x=%ld, y=%ld\n", i, pt[i].x, pt[i].y );
        j = nn[i][oct];
        printf( "nn=%ld, x=%ld, y=%ld, dist = %ld\n", j, pt[j].x, pt[j].y,
                 dist(pt[i], pt[j ]) );          
        }
//        assert( octant(pt[i], pt[ nn[i][oct] ]) == oct );

        assert( octant(pt[i], pt[ nn1[i][oct] ]) == oct );

        if( dist(pt[i], pt[ nn[i][oct] ]) != 
                dist(pt[i], pt[ nn1[i][oct] ]) ) 
       {
        printf( "NNs DON'T MATCH!\noct=%ld\n", oct );
        printf( "i=%ld, x=%ld, y=%ld\n", i, pt[i].x, pt[i].y );
        j = nn[i][oct];
        printf( "nn=%ld, x=%ld, y=%ld, dist = %ld\n", j, pt[j].x, pt[j].y,
                 dist(pt[i], pt[j ]) );
        j = nn1[i][oct];
        printf( "nn1=%ld, x=%ld, y=%ld, dist = %ld\n", j, pt[j].x, pt[j].y,
                 dist(pt[i], pt[ j ]) );
       }
//        assert( dist(pt[i], pt[ nn[i][oct] ]) == 
//                dist(pt[i], pt[ nn1[i][oct] ]) );
      }
    }
  }
  
  free( nn1 );
}

/***************************************************************************/
/***************************************************************************/

