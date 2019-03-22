#include <stdio.h>
#include <stdlib.h>

/**************************************************************************/
/*
  print error message and continue
*/

void  err_msg(
char* msg
)
{
  fprintf(stderr, "%s\n", msg);
}

/**************************************************************************/
/*
  print error message and  exit
*/

void  err_exit(
char* msg
)
{
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

