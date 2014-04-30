/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014
*/

#include "common_magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Used by chk() macro to print error message.
*/
void chk_helper( int err, const char* func, const char* file, int line )
{
    if ( err != 0 ) {
        printf( "error in %s at %s:%d: %s (%d)\n",
                func, file, line, magma_strerror(err), err );
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: ipiv(i) indicates that row i has been swapped with
      ipiv(i) from top to bottom. This function rearranges ipiv into newipiv
      where row i has to be moved to newipiv(i). The new pivoting allows for
      parallel processing vs the original one assumes a specific ordering and
      has to be done sequentially.
*/
extern "C"
void swp2pswp( int trans, int n, int *ipiv, int *newipiv){
  int i, newind, ind;

  for(i=0; i<n; i++)
    newipiv[i] = -1;
  
  if ( trans == MagmaNoTrans ){
    for(i=0; i<n; i++){
      newind = ipiv[i] - 1;
      if (newipiv[newind] == -1) {
        if (newipiv[i]==-1){
          newipiv[i] = newind;
          if (newind>i)
            newipiv[newind]= i;
        }
        else
          {
            ind = newipiv[i];
            newipiv[i] = newind;
            if (newind>i)
              newipiv[newind]= ind;
          }
      }
      else {
        if (newipiv[i]==-1){
          if (newind>i){
            ind = newipiv[newind];
            newipiv[newind] = i;
            newipiv[i] = ind;
          }
          else
            newipiv[i] = newipiv[newind];
        }
        else{
          ind = newipiv[i];
          newipiv[i] = newipiv[newind];
          if (newind > i)
            newipiv[newind] = ind;
        }
      }
    }
  } else {
    for(i=n-1; i>=0; i--){
      newind = ipiv[i] - 1;
      if (newipiv[newind] == -1) {
        if (newipiv[i]==-1){
          newipiv[i] = newind;
          if (newind>i)
            newipiv[newind]= i;
        }
        else
          {
            ind = newipiv[i];
            newipiv[i] = newind;
            if (newind>i)
              newipiv[newind]= ind;
          }
      }
      else {
        if (newipiv[i]==-1){
          if (newind>i){
            ind = newipiv[newind];
            newipiv[newind] = i;
            newipiv[i] = ind;
          }
          else
            newipiv[i] = newipiv[newind];
        }
        else{
          ind = newipiv[i];
          newipiv[i] = newipiv[newind];
          if (newind > i)
            newipiv[newind] = ind;
        }
      }
    }
  }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function sp_cat
*/
extern "C"
int sp_cat(char *lp, char *rpp[], magma_int_t *rnp, magma_int_t*np, magma_int_t ll)
{
  magma_int_t i, n, nc;
  char *f__rp;

  n = (int)*np;
  for(i = 0 ; i < n ; ++i)
    {
      nc = ll;
      if(rnp[i] < nc)
        nc = rnp[i];
      ll -= nc;
      f__rp = rpp[i];
      while(--nc >= 0)
        *lp++ = *f__rp++;
    }
  while(--ll >= 0)
    *lp++ = ' ';

  return 0;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function magma_cabs
*/
extern "C"
double magma_cabs(magmaDoubleComplex z)
{
    double __x = z.x;
    double __y = z.y;

    double __s = max(abs(__x), abs(__y));
    if(__s == 0.0)
        return __s;
    __x /= __s;
    __y /= __s;
    return __s * sqrt(__x * __x + __y * __y);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function magma_cabsf
*/
extern "C"
float magma_cabsf(magmaFloatComplex z)
{
    float __x = z.x;
    float __y = z.y;

    float __s = max(abs(__x), abs(__y));
    if(__s == 0.0)
        return __s;
    __x /= __s;
    __y /= __s;
    return __s * sqrt(__x * __x + __y * __y);
}
