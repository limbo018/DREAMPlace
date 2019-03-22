#include "global.h"

/*********************************************************************/
/*
   Return the Manhattan distance between two points
*/

long  dist(
  Point  p,
  Point  q
)
{
  long  dx, dy;
    
  dx = (p.x) - (q.x);
  if( dx < 0 )  dx = -dx;
  dy = (p.y) - (q.y);
  if( dy < 0 )  dy = -dy;

  return  dx + dy; 
}

/*********************************************************************/
/*
   Return the Manhattan distance between two points
*/

long  dist2(
  Point*  p,
  Point*  q
)
{
  long  dx, dy;
    
  dx = (p->x) - (q->x);
  if( dx < 0 )  dx = -dx;
  dy = (p->y) - (q->y);
  if( dy < 0 )  dy = -dy;

  return  dx + dy; 
}

/*********************************************************************/
/*********************************************************************/
