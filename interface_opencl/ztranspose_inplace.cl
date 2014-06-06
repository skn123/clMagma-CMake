/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @precisions normal z -> s d c

*/
// #include "common_magma.h"
// #define PRECISION_z
// #include "commonblas.h"
#include "clVendor.h"

#define PRECISION_z
//#define NB 16
#define ZSIZE_2SHARED 16

//#define NBhalf (NB/2)
#define NBhalf (ZSIZE_2SHARED/2)

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double2 magmaDoubleComplex;
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


__kernel void ztranspose_inplace_even_kernel( __global magmaDoubleComplex *matrix, int offset, int lda, int half )
{
    //__local magmaDoubleComplex a[NB][NB+1];
    //__local magmaDoubleComplex b[NB][NB+1];

    __local magmaDoubleComplex a[ZSIZE_2SHARED][ZSIZE_2SHARED+1];
    __local magmaDoubleComplex b[ZSIZE_2SHARED][ZSIZE_2SHARED+1];

    int inx = get_local_id(0);
    int iny = get_local_id(1);

    bool bottom = ( get_group_id(0) > get_group_id(1) );
    int ibx = bottom ? (get_group_id(0) - 1) : (get_group_id(1) + half);
    int iby = bottom ? (get_group_id(1))     : (get_group_id(0) + half);

    //ibx *= NB;
    //iby *= NB;

    ibx *= ZSIZE_2SHARED;
    iby *= ZSIZE_2SHARED;

    matrix += offset;

    __global magmaDoubleComplex *A = matrix + ibx + inx + (iby + iny)*lda;
    COPY_1D_TO_2D( A, lda, a, inx, iny);

    if( ibx == iby )
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        COPY_2D_TO_1D( a, inx, iny, A, lda);
    }
    else
    {
        __global magmaDoubleComplex *B = matrix + iby + inx + (ibx + iny)*lda;

        COPY_1D_TO_2D( B, lda, b, inx, iny);
        barrier(CLK_LOCAL_MEM_FENCE);

        COPY_2D_TO_1D( b, inx, iny, A, lda);
        COPY_2D_TO_1D( a, inx, iny, B, lda);
    }
}

__kernel void ztranspose_inplace_odd_kernel( __global magmaDoubleComplex *matrix, int offset, int lda, int half )
{
    //__local magmaDoubleComplex a[NB][NB+1];
    //__local magmaDoubleComplex b[NB][NB+1];
    
    __local magmaDoubleComplex a[ZSIZE_2SHARED][ZSIZE_2SHARED+1];
    __local magmaDoubleComplex b[ZSIZE_2SHARED][ZSIZE_2SHARED+1];
    
    int inx = get_local_id(0);
    int iny = get_local_id(1);
    
    bool bottom = ( get_group_id(0) >= get_group_id(1) );
    int ibx = bottom ? get_group_id(0)  : (get_group_id(1) + half - 1);
    int iby = bottom ? get_group_id(1)  : (get_group_id(0) + half);
    
    //ibx *= NB;
    //iby *= NB;
    
    ibx *= ZSIZE_2SHARED;
    iby *= ZSIZE_2SHARED;
    
    matrix += offset;
    
    __global magmaDoubleComplex *A = matrix + ibx + inx + (iby + iny)*lda;
    
    COPY_1D_TO_2D( A, lda, a, inx, iny);
    
    if( ibx == iby )
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        COPY_2D_TO_1D( a, inx, iny, A, lda);
    }
    else
    {
        __global magmaDoubleComplex *B = matrix + iby + inx + (ibx + iny)*lda;
        
        COPY_1D_TO_2D( B, lda, b, inx, iny);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        COPY_2D_TO_1D( b, inx, iny, A, lda);
        COPY_2D_TO_1D( a, inx, iny, B, lda);
    }
}
