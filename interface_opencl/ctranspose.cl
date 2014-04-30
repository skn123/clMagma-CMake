/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from ztranspose.cl normal z -> c, Fri Jan 10 15:51:19 2014
*/

//#include "common_magma.h"
//#include "commonblas.h"

#define PRECISION_c

#define CSIZE_1SHARED 32

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef float2 magmaFloatComplex;
#endif

__kernel void ctranspose_32( __global magmaFloatComplex *B, int offsetB, int ldb, __global magmaFloatComplex *A, int offsetA,  int lda )
{
    __local magmaFloatComplex a[32][CSIZE_1SHARED+1];
    
    int inx = get_local_id(0);
    int iny = get_local_id(1);
    int ibx = get_group_id(0)*32;
    int iby = get_group_id(1)*32;

    A += offsetA;
    B += offsetB;
    
    A += ibx + inx + (iby + iny)*lda;
    B += iby + inx + (ibx + iny)*ldb;
    
    a[iny+0][inx] = A[0*lda];
    a[iny+8][inx] = A[8*lda];
    a[iny+16][inx] = A[16*lda];
    a[iny+24][inx] = A[24*lda];
    
    barrier(CLK_LOCAL_MEM_FENCE);

#if defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) || defined(PRECISION_z)
    B[ 0*ldb] = a[inx][iny+ 0];
    B[ 8*ldb] = a[inx][iny+ 8];
    B[16*ldb] = a[inx][iny+16];
    B[24*ldb] = a[inx][iny+24];
#else
    B[0*ldb]    = a[inx   ][iny+0];
    B[8*ldb]    = a[inx   ][iny+8];
    B[0*ldb+16] = a[inx+16][iny+0];
    B[8*ldb+16] = a[inx+16][iny+8];

    barrier(CLK_LOCAL_MEM_FENCE);
    A += CSIZE_1SHARED;
    B += 16*ldb;

    a[iny+ 0][inx] = A[ 0*lda];
    a[iny+ 8][inx] = A[ 8*lda];
    a[iny+16][inx] = A[16*lda];
    a[iny+24][inx] = A[24*lda];

    barrier(CLK_LOCAL_MEM_FENCE);
    B[0*ldb]    = a[inx   ][iny+0];
    B[8*ldb]    = a[inx   ][iny+8];
    B[0*ldb+16] = a[inx+16][iny+0];
    B[8*ldb+16] = a[inx+16][iny+8];
#endif
}
