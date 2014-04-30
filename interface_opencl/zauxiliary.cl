/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @precisions normal z -> s d c
 */

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from zgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st
      32x32 block of A.
*/

#define PRECISION_z
#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double2 magmaDoubleComplex;
#endif

#define zlaset_threads 64

__kernel void zset_nbxnb_to_zero(int nb, __global magmaDoubleComplex *A, int offset, int lda)
{
    //int ind = blockIdx.x*lda + threadIdx.x, i, j;
    int ind = get_group_id(0)*lda+get_local_id(0);
    int i, j;
    A += (ind+offset);
    magmaDoubleComplex MAGMA_Z_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_Z_ZERO = (double2)(0.0, 0.0);
#else
    MAGMA_Z_ZERO = 0.0;
#endif
    for(i=0; i<nb; i+=32){
        for(j=0; j<nb; j+=32)
            A[j] = MAGMA_Z_ZERO;
        A += 32*lda;
    }
}

__kernel void zlaset_upper(int m, int n, __global magmaDoubleComplex *A, int offset, int lda)
{
    //int ibx = blockIdx.x * zlaset_threads;
    int ibx = get_group_id(0)*zlaset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    magmaDoubleComplex MAGMA_Z_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_Z_ZERO = (double2)(0.0, 0.0);
#else
    MAGMA_Z_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++)
        if (iby+i < n && ind < m && ind < i+iby)
            A[i*lda] = MAGMA_Z_ZERO;
}

__kernel void zlaset_lower(int m, int n, __global magmaDoubleComplex *A, int offset, int lda)
{
    //int ibx = blockIdx.x * zlaset_threads;
    int ibx = get_group_id(0)*zlaset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    magmaDoubleComplex MAGMA_Z_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_Z_ZERO = (double2)(0.0, 0.0);
#else
    MAGMA_Z_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++){
        if (iby+i < n && ind < m && ind > i+iby)
            A[i*lda] = MAGMA_Z_ZERO;
    }
}

__kernel void zlaset(int m, int n, __global magmaDoubleComplex *A, int offset, int lda)
{
    //int ibx = blockIdx.x * zlaset_threads;
    int ibx = get_group_id(0)*zlaset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    magmaDoubleComplex MAGMA_Z_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_Z_ZERO = (double2)(0.0, 0.0);
#else
    MAGMA_Z_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++)
        if (iby+i < n && ind < m)
            A[i*lda] = MAGMA_Z_ZERO;
}
