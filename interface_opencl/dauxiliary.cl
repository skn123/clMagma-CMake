/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @generated from zauxiliary.cl normal z -> d, Fri Jan 10 15:51:19 2014
 */

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from dgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st
      32x32 block of A.
*/

#define PRECISION_d
#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double double;
#endif

#define dlaset_threads 64

__kernel void dset_nbxnb_to_zero(int nb, __global double *A, int offset, int lda)
{
    //int ind = blockIdx.x*lda + threadIdx.x, i, j;
    int ind = get_group_id(0)*lda+get_local_id(0);
    int i, j;
    A += (ind+offset);
    double MAGMA_D_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_D_ZERO = (double)(0.0, 0.0);
#else
    MAGMA_D_ZERO = 0.0;
#endif
    for(i=0; i<nb; i+=32){
        for(j=0; j<nb; j+=32)
            A[j] = MAGMA_D_ZERO;
        A += 32*lda;
    }
}

__kernel void dlaset_upper(int m, int n, __global double *A, int offset, int lda)
{
    //int ibx = blockIdx.x * dlaset_threads;
    int ibx = get_group_id(0)*dlaset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    double MAGMA_D_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_D_ZERO = (double)(0.0, 0.0);
#else
    MAGMA_D_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++)
        if (iby+i < n && ind < m && ind < i+iby)
            A[i*lda] = MAGMA_D_ZERO;
}

__kernel void dlaset_lower(int m, int n, __global double *A, int offset, int lda)
{
    //int ibx = blockIdx.x * dlaset_threads;
    int ibx = get_group_id(0)*dlaset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    double MAGMA_D_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_D_ZERO = (double)(0.0, 0.0);
#else
    MAGMA_D_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++){
        if (iby+i < n && ind < m && ind > i+iby)
            A[i*lda] = MAGMA_D_ZERO;
    }
}

__kernel void dlaset(int m, int n, __global double *A, int offset, int lda)
{
    //int ibx = blockIdx.x * dlaset_threads;
    int ibx = get_group_id(0)*dlaset_threads;
    //int iby = blockIdx.y * 32;
    int iby = get_group_id(1)*32;

    //int ind = ibx + threadIdx.x;
    int ind = ibx + get_local_id(0);
    A += offset + ind + iby*lda;
    double MAGMA_D_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
    MAGMA_D_ZERO = (double)(0.0, 0.0);
#else
    MAGMA_D_ZERO = 0.0;
#endif
    for(int i=0; i<32; i++)
        if (iby+i < n && ind < m)
            A[i*lda] = MAGMA_D_ZERO;
}
