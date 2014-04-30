/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @precisions mixed zc -> ds
*/

#include "kernels_header.h"
#define PRECISION_c

__kernel void 
clag2z_generic(int M, int N, 
               __global magmaFloatComplex *SA, int SA_offset, int LDSA, 
               __global magmaDoubleComplex *A, int A_offset, int LDA ) 
{ 
    SA += SA_offset;
    A += A_offset;

    //int ibx = blockIdx.x * 64;
    int ibx = get_group_id(0) * 64;

    //int tx = threadIdx.x;
    int tx = get_local_id(0);
    //int ty = threadIdx.y;
    int ty = get_local_id(1);
    int idt = ty * 16 + tx;
        
    if( (ibx+idt) >= M ){
        SA += (M-1);
        A  += (M-1);
    }
    else{
        SA += ibx+idt;
        A  += ibx+idt;
    }
    __global magmaFloatComplex * SAend = SA+LDSA*N;
    //magmaDoubleComplex Ap[1]={ cuComplexFloatToDouble(SA[0]) };
    magmaDoubleComplex Ap[1];
    MAGMA_Z_REAL(Ap[0]) = (double)MAGMA_C_REAL(SA[0]);
#if defined(PRECISION_z) || defined(PRECISION_c)
    MAGMA_Z_IMAG(Ap[0]) = (double)MAGMA_C_IMAG(SA[0]);
#endif

    do {
        SA  += LDSA;
        A[0] = Ap[0];
        //Ap[0]= cuComplexFloatToDouble(SA[0]);
        MAGMA_Z_REAL(Ap[0]) = (double)MAGMA_C_REAL(SA[0]);
#if defined(PRECISION_z) || defined(PRECISION_c)
        MAGMA_Z_IMAG(Ap[0]) = (double)MAGMA_C_IMAG(SA[0]);
#endif
        A   += LDA;

    } while (SA < SAend);

    A[0] = Ap[0];
}

__kernel void 
clag2z_special(int M, int N, 
               __global magmaFloatComplex *SA, int SA_offset, int LDSA, 
               __global magmaDoubleComplex *A, int A_offset,  int LDA ) 
{ 
    SA += SA_offset;
    A += A_offset;
    
    //int ibx = blockIdx.x * 64;
    int ibx = get_group_id(0) * 64;

    //int tx = threadIdx.x;
    int tx = get_local_id(0);
    //int ty = threadIdx.y;
    int ty = get_local_id(1);
    int idt = ty * 16 + tx;
        
    if( (ibx+idt) >= M ){
        SA += (M-1);
        A  += (M-1);
    }
    else{
        SA += ibx+idt;
        A  += ibx+idt;
    }
    //magmaDoubleComplex Ap[1] = { cuComplexFloatToDouble(SA[0]) };
    magmaDoubleComplex Ap[1];
    MAGMA_Z_REAL(Ap[0]) = (double)MAGMA_C_REAL(SA[0]);
#if defined(PRECISION_z) || defined(PRECISION_c)
    MAGMA_Z_IMAG(Ap[0]) = (double)MAGMA_C_IMAG(SA[0]);
#endif
    A[0] = Ap[0];
}


