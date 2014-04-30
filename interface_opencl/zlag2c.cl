/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @precisions mixed zc -> ds

*/
#include "kernels_header.h"
#define PRECISION_z
#define blksize 64

//magmaint_zlag2c
__kernel void 
magmaint_zlag2c(int M, int N, __global magmaDoubleComplex *A, int A_offset, int lda, 
                __global magmaFloatComplex *SA, int SA_offset, int ldsa,
                double RMAX)
{
    A += A_offset;
    SA += SA_offset;
    
    __global magmaDoubleComplex *Aend = A + lda*N;
    magmaDoubleComplex tmp;
    double mRMAX = - RMAX;
    int    mym   = get_group_id(0) * blksize + get_local_id(0);

    if ( mym < M ){
        A += mym;
        SA+= mym; 
        
        tmp = *A;
        for ( ; A < Aend; )
        {
            A  += lda;
            if(    ( MAGMA_Z_REAL(tmp) < mRMAX) || (MAGMA_Z_REAL(tmp) > RMAX)
#if defined(PRECISION_z) || defined(PRECISION_c)
                || (MAGMA_Z_IMAG(tmp) < mRMAX) || (MAGMA_Z_IMAG(tmp) > RMAX) 
#endif
                )
            {
                // flag is not used in opencl version
                //flag = 1; 
            }
            //*SA = cuComplexDoubleToFloat( tmp );
            MAGMA_Z_REAL(*SA) = (float)MAGMA_Z_REAL(tmp);
#if defined(PRECISION_z) || defined(PRECISION_c)
            MAGMA_Z_IMAG(*SA) = (float)MAGMA_Z_IMAG(tmp);
#endif
            
            tmp = *A;
            SA += ldsa;
        }
    }
}

