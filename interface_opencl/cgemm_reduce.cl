/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zgemm_reduce.cl normal z -> c, Fri Jan 10 15:51:19 2014
*/

#include "kernels_header.h"


//#define NUM_THREADS 1024
#define NUM_THREADS 256

///////////////////////////////////////////////////////////////////////////////////////////////////
// size of work for a thread block
#define BLK_M 8
#define BLK_N 8

#define BLK_K (NUM_THREADS / (BLK_M * BLK_N))

///////////////////////////////////////////////////////////////////////////////////////////////////
// ----------------------------------------
// Does sum reduction of array x, leaving total in x[0].
// Contents of x are destroyed in the process.
// With k threads, can reduce array up to 2*k in size.
// Assumes number of threads <= 1024
// Having n as template parameter allows compiler to evaluate some conditions at compile time.

void sum_reduce2( int n, int j, int k, int i, __local magmaFloatComplex x[][ BLK_N +1][ BLK_K +1] )
{
    barrier(CLK_LOCAL_MEM_FENCE);
/*
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[j][k][i] += x[j][k][i+1024]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[j][k][i] += x[j][k][i+ 512]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[j][k][i] += x[j][k][i+ 256]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[j][k][i] += x[j][k][i+ 128]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[j][k][i] += x[j][k][i+  64]; }  barrier(CLK_LOCAL_MEM_FENCE); }
*/
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[j][k][i] += x[j][k][i+  32]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[j][k][i] += x[j][k][i+  16]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[j][k][i] += x[j][k][i+   8]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[j][k][i] += x[j][k][i+   4]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[j][k][i] += x[j][k][i+   2]; }  barrier(CLK_LOCAL_MEM_FENCE); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[j][k][i] += x[j][k][i+   1]; }  barrier(CLK_LOCAL_MEM_FENCE); }
}
// end sum_reduce


//==============================================================================

__kernel 
void magmablas_cgemm_reduce_kernel(int k, magmaFloatComplex alpha, 
                                   __global magmaFloatComplex *d_A, int d_A_offset, int lda,
                                   __global magmaFloatComplex *d_B, int d_B_offset, int ldb,
                                   magmaFloatComplex beta,
                                   __global magmaFloatComplex *d_C, int d_C_offset, int ldc)
{
        d_A += d_A_offset;
        d_B += d_B_offset;
        d_C += d_C_offset;

        const int i = get_local_id(0);

        /*
        const magmaFloatComplex *dA = d_A + (blockIdx.x*BLK_M + threadIdx.y) * lda;
        const magmaFloatComplex *dB = d_B + (blockIdx.y*BLK_N + threadIdx.z) * ldb;
        magmaFloatComplex *dC = d_C + blockIdx.x*BLK_M + blockIdx.y*BLK_N * ldc;
        */

        d_A += (get_group_id(0)*BLK_M + get_local_id(1)) * lda;
        d_B += (get_group_id(1)*BLK_N + get_local_id(2)) * ldb;
        d_C += get_group_id(0)*BLK_M + get_group_id(1)*BLK_N * ldc;

        __local magmaFloatComplex sum[BLK_M][BLK_N+1][ BLK_K +1];
        magmaFloatComplex lsum;

        /*  w := v' * C  */
        lsum = MAGMA_C_ZERO;
        for( int j = i; j < k; j += BLK_K )
            lsum += MAGMA_C_CNJG( d_A[j] )* d_B[j];
        
        sum[get_local_id(1)][get_local_id(2)][i] = lsum;
        sum_reduce2(BLK_K,  get_local_id(1), get_local_id(2), i, sum );

        /*  C := C - v * w  */
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) == 0)
           if (MAGMA_C_EQUAL(beta, MAGMA_C_ZERO))
              d_C[get_local_id(1) + get_local_id(2)*ldc] = alpha*sum[get_local_id(1)][get_local_id(2)][0];
           else
              d_C[get_local_id(1) + get_local_id(2)*ldc] = beta* d_C[get_local_id(1) + get_local_id(2)*ldc] + 
                                                  alpha*sum[get_local_id(1)][get_local_id(2)][0];
}
//==============================================================================



