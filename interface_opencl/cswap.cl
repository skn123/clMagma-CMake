/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @generated from zswap.cl normal z -> c, Fri Jan 10 15:51:19 2014
 */

#define PRECISION_c
#define BLOCK_SIZE 64

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef float2 magmaFloatComplex;
#endif

typedef struct {
    int n, offset_dA1, lda1, offset_dA2, lda2;
} magmagpu_cswap_params_t;

__kernel void magmagpu_cswap(__global magmaFloatComplex *dA1, __global magmaFloatComplex *dA2, magmagpu_cswap_params_t params )
{
    unsigned int x = get_local_id(0) + get_local_size(0)*get_group_id(0);
    unsigned int offset1 = x*params.lda1;
    unsigned int offset2 = x*params.lda2;
    if( x < params.n ){
        __global magmaFloatComplex *A1  = dA1 + params.offset_dA1 + offset1;
        __global magmaFloatComplex *A2  = dA2 + params.offset_dA2 + offset2;
        magmaFloatComplex temp = *A1;
        *A1 = *A2;
        *A2 = temp;
    }
}

// empty kernel, benchmark in iwocl 2013
__kernel void cswap_empty_kernel(int i0, int i1, int i2, int i3, int i4, 
                                 int i5, int i6, int i7, int i8, int i9,
                                 float d0, float d1, float d2, float d3, float d4, 
                                 __global float *dA, __global float *dB, __global float *dC)
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

