/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from ztranspose_inplace.cl normal z -> c, Fri Jan 10 15:51:19 2014

*/
// #include "common_magma.h"
// #define PRECISION_c
// #include "commonblas.h"
#include "clVendor.h"

#define PRECISION_c
//#define NB 16
#define CSIZE_2SHARED 16

//#define NBhalf (NB/2)
#define NBhalf (CSIZE_2SHARED/2)

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef float2 magmaFloatComplex;
#endif

//#if NB == 32
#define COPY_1D_TO_2D( a, lda, b, inx, iny )   \
    b[iny][inx]        = a[0];                 \
    b[iny+NBhalf][inx] = a[NBhalf*lda];

#define COPY_2D_TO_1D( b, inx, iny, a, lda )   \
    a[0]               = b[inx][iny];          \
    a[NBhalf*lda]      = b[inx][iny+NBhalf];

/*
#else

#define COPY_1D_TO_2D( a, lda, b, inx, iny )   \
    b[iny][inx] = a[0];

#define COPY_2D_TO_1D( b, inx, iny, a, lda )   \
    a[0]        = b[inx][iny];

#endif
*/


__kernel void ctranspose_inplace_even_kernel( __global magmaFloatComplex *matrix, int offset, int lda, int half )
{
    //__local magmaFloatComplex a[NB][NB+1];
    //__local magmaFloatComplex b[NB][NB+1];

    __local magmaFloatComplex a[CSIZE_2SHARED][CSIZE_2SHARED+1];
    __local magmaFloatComplex b[CSIZE_2SHARED][CSIZE_2SHARED+1];

    int inx = get_local_id(0);
    int iny = get_local_id(1);

    bool bottom = ( get_group_id(0) > get_group_id(1) );
    int ibx = bottom ? (get_group_id(0) - 1) : (get_group_id(1) + half);
    int iby = bottom ? (get_group_id(1))     : (get_group_id(0) + half);

    //ibx *= NB;
    //iby *= NB;

    ibx *= CSIZE_2SHARED;
    iby *= CSIZE_2SHARED;

    matrix += offset;

    __global magmaFloatComplex *A = matrix + ibx + inx + (iby + iny)*lda;
    COPY_1D_TO_2D( A, lda, a, inx, iny);

    if( ibx == iby )
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        COPY_2D_TO_1D( a, inx, iny, A, lda);
    }
    else
    {
        __global magmaFloatComplex *B = matrix + iby + inx + (ibx + iny)*lda;

        COPY_1D_TO_2D( B, lda, b, inx, iny);
        barrier(CLK_LOCAL_MEM_FENCE);

        COPY_2D_TO_1D( b, inx, iny, A, lda);
        COPY_2D_TO_1D( a, inx, iny, B, lda);
    }
}

__kernel void ctranspose_inplace_odd_kernel( __global magmaFloatComplex *matrix, int offset, int lda, int half )
{
    //__local magmaFloatComplex a[NB][NB+1];
    //__local magmaFloatComplex b[NB][NB+1];
    
    __local magmaFloatComplex a[CSIZE_2SHARED][CSIZE_2SHARED+1];
    __local magmaFloatComplex b[CSIZE_2SHARED][CSIZE_2SHARED+1];
    
    int inx = get_local_id(0);
    int iny = get_local_id(1);
    
    bool bottom = ( get_group_id(0) >= get_group_id(1) );
    int ibx = bottom ? get_group_id(0)  : (get_group_id(1) + half - 1);
    int iby = bottom ? get_group_id(1)  : (get_group_id(0) + half);
    
    //ibx *= NB;
    //iby *= NB;
    
    ibx *= CSIZE_2SHARED;
    iby *= CSIZE_2SHARED;
    
    matrix += offset;
    
    __global magmaFloatComplex *A = matrix + ibx + inx + (iby + iny)*lda;
    
    COPY_1D_TO_2D( A, lda, a, inx, iny);
    
    if( ibx == iby )
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        COPY_2D_TO_1D( a, inx, iny, A, lda);
    }
    else
    {
        __global magmaFloatComplex *B = matrix + iby + inx + (ibx + iny)*lda;
        
        COPY_1D_TO_2D( B, lda, b, inx, iny);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        COPY_2D_TO_1D( b, inx, iny, A, lda);
        COPY_2D_TO_1D( a, inx, iny, B, lda);
    }
}
