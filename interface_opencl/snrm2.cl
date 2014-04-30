/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from dznrm2.cl normal z -> s, Fri Jan 10 15:51:19 2014
*/

#include "kernels_header.h"

/*
#define HAVE_clAmdblas
#include "magma_types.h"
*/
//#define BLOCK_SIZE  512
#define BLOCK_SIZE  256
#define BLOCK_SIZEx  32
//#define BLOCK_SIZEy  16
#define BLOCK_SIZEy  8

#define PRECISION_s
/*
#if defined(PRECISION_c) || defined(PRECISION_z)
typedef float float;
#define MAGMA_S_REAL(a)       (a).x
#define MAGMA_S_IMAG(a)       (a).y
#endif

#if defined(PRECISION_z)
#define MAGMA_S_ABS(a)        magma_fabsf(a)
#elif defined(PRECISION_c)
#define MAGMA_C_ABS(a)        magma_fabsff(a)
#endif
*/

// ----------------------------------------
// Does sum reduction of array x, leaving total in x[0].
// Contents of x are destroyed in the process.
// With k threads, can reduce array up to 2*k in size.
// Assumes number of threads <= 1024
// Having n as template parameter allows compiler to evaluate some conditions at compile time.
//template< int n > void sum_reduce( int i, float* x )
void sum_reduce(int n,  int i, __local float* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  barrier(CLK_LOCAL_MEM_FENCE); }
}
// end sum_reduce

//==============================================================================
void dsum_reduce( int n, int i, __local float* x )
{
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  barrier(CLK_LOCAL_MEM_FENCE); }
}
// end sum_reduce

//==============================================================================
__kernel void
magmablas_snrm2_kernel( int m, __global float *da, int da_offset, int ldda, __global float *dxnorm, int dxnorm_offset )
{
    da += da_offset;
    dxnorm += dxnorm_offset;
    const int i = get_local_id(0);
    
    //float *dx = da + get_group_id(0) * ldda;
    da += get_group_id(0) * ldda;

    __local float sum[ BLOCK_SIZE ];
    float re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += BLOCK_SIZE ) {

#if (defined(PRECISION_s) || defined(PRECISION_d))
//        re = dx[j];
        re = da[j];
        lsum += re*re;
#else
//        re = MAGMA_S_REAL( dx[j] );
//        float im = MAGMA_S_IMAG( dx[j] );
        re = MAGMA_S_REAL( da[j] );
        float im = MAGMA_S_IMAG( da[j] );
        lsum += re*re + im*im;
#endif

    }
    sum[i] = lsum;
//    sum_reduce< BLOCK_SIZE >( i, sum );
    sum_reduce(BLOCK_SIZE, i, sum );
    
    if (i==0)
       dxnorm[get_group_id(0)] = sqrt(sum[0]);
}

__kernel void
magmablas_snrm2_adjust_kernel(__global float *xnorm, int xnorm_offset, __global float *c, int c_offset)
{
   xnorm += xnorm_offset;
   c += c_offset;
    
    const int i = get_local_id(0);

   __local float sum[ BLOCK_SIZE ];
   float temp;

   temp = MAGMA_S_ABS( c[i] ) / xnorm[0];
   sum[i] = -temp * temp;
   dsum_reduce( get_local_size(0), i, sum );

   barrier(CLK_LOCAL_MEM_FENCE);
   
   if (i==0)
     xnorm[0] = xnorm[0] * sqrt(1+sum[0]);
}

