/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @precisions normal z -> s d c
 */

#define PRECISION_z
#define BLOCK_SIZE 64

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double2 magmaDoubleComplex;
#endif

typedef struct {
    int n, offset_dA1, lda1, offset_dA2, lda2;
} magmagpu_zswap_params_t;

__kernel void magmagpu_zswap(__global magmaDoubleComplex *dA1, __global magmaDoubleComplex *dA2, magmagpu_zswap_params_t params )
{
    unsigned int x = get_local_id(0) + get_local_size(0)*get_group_id(0);
    unsigned int offset1 = x*params.lda1;
    unsigned int offset2 = x*params.lda2;
    if( x < params.n ){
        __global magmaDoubleComplex *A1  = dA1 + params.offset_dA1 + offset1;
        __global magmaDoubleComplex *A2  = dA2 + params.offset_dA2 + offset2;
        magmaDoubleComplex temp = *A1;
        *A1 = *A2;
        *A2 = temp;
    }
}

// empty kernel, benchmark in iwocl 2013
__kernel void zswap_empty_kernel(int i0, int i1, int i2, int i3, int i4, 
                                 int i5, int i6, int i7, int i8, int i9,
                                 double d0, double d1, double d2, double d3, double d4, 
                                 __global double *dA, __global double *dB, __global double *dC)
{
    int x = get_local_id(0);

    for(int i=0;i<i0;i++)
    {
        dC[i+x] += d1*dC[i+x] + d2*dA[i]*dB[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i=0;i<i0;i++)
    {
        dC[i+x] += d1*dC[i+x] + d2*dA[i]*dB[i];
    }
}

