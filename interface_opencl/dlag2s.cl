/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zlag2c.cl mixed zc -> ds, Fri Jan 10 15:51:19 2014

*/
#include "kernels_header.h"
#define PRECISION_d
#define blksize 64

//magmaint_dlag2s
__kernel void 
magmaint_dlag2s(int M, int N, __global double *A, int A_offset, int lda, 
                __global float *SA, int SA_offset, int ldsa,
                double RMAX)
{
    A += A_offset;
    SA += SA_offset;
    
    __global double *Aend = A + lda*N;
    double tmp;
    double mRMAX = - RMAX;
    int    mym   = get_group_id(0) * blksize + get_local_id(0);

    if ( mym < M ){
        A += mym;
        SA+= mym; 
        
        tmp = *A;
        for ( ; A < Aend; )
        {
            A  += lda;
            if(    ( MAGMA_D_REAL(tmp) < mRMAX) || (MAGMA_D_REAL(tmp) > RMAX)
#if defined(PRECISION_z) || defined(PRECISION_c)
                || (MAGMA_D_IMAG(tmp) < mRMAX) || (MAGMA_D_IMAG(tmp) > RMAX) 
#endif
                )
            {
                // flag is not used in opencl version
                //flag = 1; 
            }
            //*SA = (float)( tmp );
            MAGMA_D_REAL(*SA) = (float)MAGMA_D_REAL(tmp);
#if defined(PRECISION_z) || defined(PRECISION_c)
            MAGMA_D_IMAG(*SA) = (float)MAGMA_D_IMAG(tmp);
#endif
            
            tmp = *A;
            SA += ldsa;
        }
    }
}

