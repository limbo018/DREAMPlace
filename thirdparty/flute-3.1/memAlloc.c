/* --------------------------------------------------------------------------
   Public domain memory allocation and de-allocation routines.
   Taken from Appendix B of: 
   Numerical Recipes in C: The Art of Scientific Computing, Second Edition,
   Cambridge University Press, 1992
----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#include "memAlloc.h"

#define MEM_END 1
#define FREE_ARG char*

void runtimeError(char error_text[])
/* error handler */
{
  fprintf(stderr, "ERROR: %s \n", error_text);
  fprintf(stderr, "Aborting !! \n");
  fflush(stdout);
  fflush(stderr);
  exit(1);
}

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
  float *v;
  v=(float *)malloc((size_t) ((nh-nl+1+MEM_END)*sizeof(float)));
  if (!v) runtimeError("allocation failure in vector()");
  return v-nl+MEM_END;
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
  int *v;
  v=(int *)malloc((size_t) ((nh-nl+1+MEM_END)*sizeof(int)));
  if (!v) runtimeError("allocation failure in ivector()");
  return v-nl+MEM_END;
}

unsigned char *cvector(long nl, long nh)
/* allocate an unsigned char vector with subscript range v[nl..nh] */
{
  unsigned char *v;
  v=(unsigned char *)malloc((size_t) ((nh-nl+1+MEM_END)*sizeof(unsigned char)));
  if (!v) runtimeError("allocation failure in cvector()");
  return v-nl+MEM_END;
}

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
  unsigned long *v;
  v=(unsigned long *)malloc((size_t) ((nh-nl+1+MEM_END)*sizeof(long)));
  if (!v) runtimeError("allocation failure in lvector()");
  return v-nl+MEM_END;
}

double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
  double *v;
  v=(double *)malloc((size_t) ((nh-nl+1+MEM_END)*sizeof(double)));
  if (!v) runtimeError("allocation failure in dvector()");
  return v-nl+MEM_END;
}

float **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  float **m;
  
  /* allocate pointers to rows */
  m=(float **) malloc((size_t)((nrow+MEM_END)*sizeof(float*)));
  if (!m) runtimeError("allocation failure 1 in matrix()");
  m += MEM_END;
  m -= nrl;
  
  /* allocate rows and set pointers to them */
  m[nrl]=(float *) malloc((size_t)((nrow*ncol+MEM_END)*sizeof(float)));
  if (!m[nrl]) runtimeError("allocation failure 2 in matrix()");
  m[nrl] += MEM_END;
  m[nrl] -= ncl;
  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
  /* return pointer to array of pointers to rows */
  return m;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  double **m;

  /* allocate pointers to rows */
  m=(double **) malloc((size_t)((nrow+MEM_END)*sizeof(double*)));
  if (!m) runtimeError("allocation failure 1 in dmatrix()");
  m += MEM_END;
  m -= nrl;
  
  /* allocate rows and set pointers to them */
  m[nrl]=(double *) malloc((size_t)((nrow*ncol+MEM_END)*sizeof(double)));
  if (!m[nrl]) runtimeError("allocation failure 2 in dmatrix()");
  m[nrl] += MEM_END;
  m[nrl] -= ncl;
  
  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
  
  /* return pointer to array of pointers to rows */
  return m;
}

int **imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  int **m;

  /* allocate pointers to rows */
  m=(int **) malloc((size_t)((nrow+MEM_END)*sizeof(int*)));
  if (!m) runtimeError("allocation failure 1 in imatrix()");
  m += MEM_END;
  m -= nrl;
  
  /* allocate rows and set pointers to them */
  m[nrl]=(int *) malloc((size_t)((nrow*ncol+MEM_END)*sizeof(int)));
  if (!m[nrl]) runtimeError("allocation failure 2 in imatrix()");
  m[nrl] += MEM_END;
  m[nrl] -= ncl;
  
  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
  
  /* return pointer to array of pointers to rows */
  return m;
}


char **cmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a char matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  char **m;

  /* allocate pointers to rows */
  m=(char **) malloc((size_t)((nrow+MEM_END)*sizeof(char*)));
  if (!m) runtimeError("allocation failure 1 in cmatrix()");
  m += MEM_END;
  m -= nrl;
  
  /* allocate rows and set pointers to them */
  m[nrl]=(char *) malloc((size_t)((nrow*ncol+MEM_END)*sizeof(char)));
  if (!m[nrl]) runtimeError("allocation failure 2 in cmatrix()");
  m[nrl] += MEM_END;
  m[nrl] -= ncl;
  
  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
  
  /* return pointer to array of pointers to rows */
  return m;
}


unsigned long **lmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  unsigned long **m;

  /* allocate pointers to rows */
  m=(unsigned long **) malloc((size_t)((nrow+MEM_END)*sizeof(long*)));
  if (!m) runtimeError("allocation failure 1 in lmatrix()");
  m += MEM_END;
  m -= nrl;
  
  /* allocate rows and set pointers to them */
  m[nrl]=(unsigned long *) malloc((size_t)((nrow*ncol+MEM_END)*sizeof(long)));
  if (!m[nrl]) runtimeError("allocation failure 2 in lmatrix()");
  m[nrl] += MEM_END;
  m[nrl] -= ncl;
  
  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
  
  /* return pointer to array of pointers to rows */
  return m;
}


float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch, 
  long newrl, long newcl)
/* point a submatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
{
  long i,j,nrow=oldrh-oldrl+1,ncol=oldcl-newcl;
  float **m;
  
  /* allocate array of pointers to rows */
  m=(float **) malloc((size_t) ((nrow+MEM_END)*sizeof(float*)));
  if (!m) runtimeError("allocation failure in submatrix()");
  m += MEM_END;
  m -= newrl;
  
  /* set pointers to rows */
  for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+ncol;
  
  /* return pointer to array of pointers to rows */
  return m;
}

float **convert_matrix(float *a, long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix m[nrl..nrh][ncl..nch] that points to the matrix
declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
and ncol=nch-ncl+1. The routine should be called with the address
&a[0][0] as the first argument. */
{
  long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1;
  float **m;

  /* allocate pointers to rows */
  m=(float **) malloc((size_t) ((nrow+MEM_END)*sizeof(float*)));
  if (!m) runtimeError("allocation failure in convert_matrix()");
  m += MEM_END;
  m -= nrl;
  
  /* set pointers to rows */
  m[nrl]=a-ncl;
  for(i=1,j=nrl+1;i<nrow;i++,j++) m[j]=m[j-1]+ncol;
  
  /* return pointer to array of pointers to rows */
  return m;
}

float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
  long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
  float ***t;

  /* allocate pointers to pointers to rows */
  t=(float ***) malloc((size_t)((nrow+MEM_END)*sizeof(float**)));
  if (!t) runtimeError("allocation failure 1 in f3tensor()");
  t += MEM_END;
  t -= nrl;
  
  /* allocate pointers to rows and set pointers to them */
  t[nrl]=(float **) malloc((size_t)((nrow*ncol+MEM_END)*sizeof(float*)));
  if (!t[nrl]) runtimeError("allocation failure 2 in f3tensor()");
  t[nrl] += MEM_END;
  t[nrl] -= ncl;
  
  /* allocate rows and set pointers to them */
  t[nrl][ncl]=(float *) malloc((size_t)((nrow*ncol*ndep+MEM_END)*sizeof(float)));
  if (!t[nrl][ncl]) runtimeError("allocation failure 3 in f3tensor()");
  t[nrl][ncl] += MEM_END;
  t[nrl][ncl] -= ndl;
  
  for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
  for(i=nrl+1;i<=nrh;i++) {
    t[i]=t[i-1]+ncol;
    t[i][ncl]=t[i-1][ncl]+ncol*ndep;
    for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
  }
  
  /* return pointer to array of pointers to rows */
  return t;
}


unsigned long ***lmatrix3D(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate an unsigned long 3D matrix with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
  long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
  unsigned long ***t;
  long d1, d2, d3;
  
  /* allocate pointers to pointers to rows */
  t=(unsigned long ***) malloc((size_t)((nrow+MEM_END)*sizeof(long**)));
  if (!t) runtimeError("allocation failure 1 in lmatrix3D()");
  t += MEM_END;
  t -= nrl;
  
  /* allocate pointers to rows and set pointers to them */
  t[nrl]=(unsigned long **) malloc((size_t)((nrow*ncol+MEM_END)*sizeof(long*)));
  if (!t[nrl]) runtimeError("allocation failure 2 in lmatrix3D()");
  t[nrl] += MEM_END;
  t[nrl] -= ncl;
  
  /* allocate rows and set pointers to them */
  t[nrl][ncl]=(unsigned long *) malloc((size_t)((nrow*ncol*ndep+MEM_END)*sizeof(long)));
  if (!t[nrl][ncl]) runtimeError("allocation failure 3 in lmatrix3D()");
  t[nrl][ncl] += MEM_END;
  t[nrl][ncl] -= ndl;
  
  for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
  for(i=nrl+1;i<=nrh;i++) {
    t[i]=t[i-1]+ncol;
    t[i][ncl]=t[i-1][ncl]+ncol*ndep;
    for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
  }  
  
  /* return pointer to array of pointers to rows */
  return t;
}


int ***imatrix3D(int nrl, int nrh, int ncl, int nch, int ndl, int ndh)
/* allocate an int 3D matrix with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
  int i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
  int ***t;
  int d1, d2, d3;
  
  /* allocate pointers to pointers to rows */
  t=(int ***) malloc((size_t)((nrow+MEM_END)*sizeof(int**)));
  if (!t) runtimeError("allocation failure 1 in imatrix3D()");
  t += MEM_END;
  t -= nrl;
  
  /* allocate pointers to rows and set pointers to them */
  t[nrl]=(int **) malloc((size_t)((nrow*ncol+MEM_END)*sizeof(int*)));
  if (!t[nrl]) runtimeError("allocation failure 2 in imatrix3D()");
  t[nrl] += MEM_END;
  t[nrl] -= ncl;
  
  /* allocate rows and set pointers to them */
  t[nrl][ncl]=(int *) malloc((size_t)((nrow*ncol*ndep+MEM_END)*sizeof(int)));
  if (!t[nrl][ncl]) runtimeError("allocation failure 3 in imatrix3D()");
  t[nrl][ncl] += MEM_END;
  t[nrl][ncl] -= ndl;
  
  for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
  for(i=nrl+1;i<=nrh;i++) {
    t[i]=t[i-1]+ncol;
    t[i][ncl]=t[i-1][ncl]+ncol*ndep;
    for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
  }  
  
  /* return pointer to array of pointers to rows */
  return t;
}


void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-MEM_END));
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
  free((FREE_ARG) (v+nl-MEM_END));
}

void free_cvector(unsigned char *v, long nl, long nh)
/* free an unsigned char vector allocated with cvector() */
{
  free((FREE_ARG) (v+nl-MEM_END));
}

void free_lvector(unsigned long *v, long nl, long nh)
/* free an unsigned long vector allocated with lvector() */
{
  free((FREE_ARG) (v+nl-MEM_END));
}

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
  free((FREE_ARG) (v+nl-MEM_END));
}

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-MEM_END));
  free((FREE_ARG) (m+nrl-MEM_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-MEM_END));
  free((FREE_ARG) (m+nrl-MEM_END));
}

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
/* free an int matrix allocated by imatrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-MEM_END));
  free((FREE_ARG) (m+nrl-MEM_END));
}

void free_cmatrix(char **m, long nrl, long nrh, long ncl, long nch)
/* free a char matrix allocated by imatrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-MEM_END));
  free((FREE_ARG) (m+nrl-MEM_END));
}

void free_lmatrix(unsigned long **m, long nrl, long nrh, long ncl, long nch)
/* free an unsigned long matrix allocated by lmatrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-MEM_END));
  free((FREE_ARG) (m+nrl-MEM_END));
}

void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a submatrix allocated by submatrix() */
{
  free((FREE_ARG) (b+nrl-MEM_END));
}

void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a matrix allocated by convert_matrix() */
{
  free((FREE_ARG) (b+nrl-MEM_END));
}

void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,
  long ndl, long ndh)
/* free a float f3tensor allocated by f3tensor() */
{
  free((FREE_ARG) (t[nrl][ncl]+ndl-MEM_END));
  free((FREE_ARG) (t[nrl]+ncl-MEM_END));
  free((FREE_ARG) (t+nrl-MEM_END));
}


void free_lmatrix3D(unsigned long ***t, long nrl, long nrh, long ncl, long nch,
  long ndl, long ndh)
/* free an unsigned long 3D matrix allocated by lmatrix3D() */
{
  free((FREE_ARG) (t[nrl][ncl]+ndl-MEM_END));
  free((FREE_ARG) (t[nrl]+ncl-MEM_END));
  free((FREE_ARG) (t+nrl-MEM_END));
}


void free_imatrix3D(int ***t, int nrl, int nrh, int ncl, int nch,
  int ndl, int ndh)
/* free an unsigned int 3D matrix allocated by imatrix3D() */
{
  free((FREE_ARG) (t[nrl][ncl]+ndl-MEM_END));
  free((FREE_ARG) (t[nrl]+ncl-MEM_END));
  free((FREE_ARG) (t+nrl-MEM_END));
}


int comp_float(const void *i, const void *j) {
    if(*(float *)i < *(float *)j)
        return -1;
    else if(*(float *)i > *(float *)j)
        return 1;
    else
        return 0;
}


int comp_int(const void *i, const void *j) {
    return *(int *)i - *(int *)j;
}
