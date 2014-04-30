/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @precisions normal z -> s d c
 */
//#include "common_magma.h"

#define PRECISION_z
#define BLOCK_SIZE 32

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double2 magmaDoubleComplex;
#endif

typedef struct {
        int n, lda, j0;
        short ipiv[BLOCK_SIZE];
} zlaswp_params_t;

typedef struct {
        int n, lda, j0, npivots;
        short ipiv[BLOCK_SIZE];
} zlaswp_params_t2;

/*
 * Old version
 */
__kernel void myzlaswp2(__global magmaDoubleComplex *Ain, int offset, zlaswp_params_t2 params)
{
    unsigned int tid = get_local_id(0) + get_local_size(0)*get_group_id(0);

    if( tid < params.n )
    {
        int lda = params.lda;
        __global magmaDoubleComplex *A = Ain + offset + tid + lda*params.j0;

        for( int i = 0; i < params.npivots; i++ )
        {
            int j = params.ipiv[i];
            __global magmaDoubleComplex *p1 = A + i*lda;
            __global magmaDoubleComplex *p2 = A + j*lda;
            magmaDoubleComplex temp = *p1;
            *p1 = *p2;
            *p2 = temp;
        }
    }
}
