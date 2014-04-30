/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @generated from zauxiliary.cl normal z -> c, Fri Jan 10 15:51:19 2014
 */

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from cgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st
      32x32 block of A.
*/

#define PRECISION_c
#if defined(PRECISION_c) || defined(PRECISION_z)
typedef float2 magmaFloatComplex;
#endif

#define claset_threads 64

__kernel void cset_nbxnb_to_zero(int nb, __global magmaFloatComplex *A, int offset, int lda)
{
    //int ind = blockIdx.x*lda + threadIdx.x, i, j;
    int ind = get_group_id(0)*lda+get_local_id(0);
    int i, j;
    A += (ind+offset);
    magmaFloatComplex MAGMA_C_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_C_ZERO = (float2)(0.0, 0.0);
#else
    MAGMA_C_ZERO = 0.0;
#endif
    for(i=0; i<nb; i+=32){
        for(j=0; j<nb; j+=32)
            A[j] = MAGMA_C_ZERO;
        A += 32*lda;
    }
}

__kernel void claset_upper(int m, int n, __global magmaFloatComplex *A, int offset, int lda)
{
    //int ibx = blockIdx.x * claset_threads;
    int ibx = get_group_id(0)*claset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    magmaFloatComplex MAGMA_C_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_C_ZERO = (float2)(0.0, 0.0);
#else
    MAGMA_C_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++)
        if (iby+i < n && ind < m && ind < i+iby)
            A[i*lda] = MAGMA_C_ZERO;
}

__kernel void claset_lower(int m, int n, __global magmaFloatComplex *A, int offset, int lda)
{
    //int ibx = blockIdx.x * claset_threads;
    int ibx = get_group_id(0)*claset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    magmaFloatComplex MAGMA_C_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_C_ZERO = (float2)(0.0, 0.0);
#else
    MAGMA_C_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++){
        if (iby+i < n && ind < m && ind > i+iby)
            A[i*lda] = MAGMA_C_ZERO;
    }
}

__kernel void claset(int m, int n, __global magmaFloatComplex *A, int offset, int lda)
{
    //int ibx = blockIdx.x * claset_threads;
    int ibx = get_group_id(0)*claset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    magmaFloatComplex MAGMA_C_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_C_ZERO = (float2)(0.0, 0.0);
#else
    MAGMA_C_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++)
        if (iby+i < n && ind < m)
            A[i*lda] = MAGMA_C_ZERO;
}
