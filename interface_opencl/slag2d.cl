/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from clag2z.cl mixed zc -> ds, Fri Jan 10 15:51:19 2014
*/

#include "kernels_header.h"
#define PRECISION_s

__kernel void 
slag2d_generic(int M, int N, 
               __global float *SA, int SA_offset, int LDSA, 
               __global double *A, int A_offset, int LDA ) 
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
    __global float * SAend = SA+LDSA*N;
    //double Ap[1]={ (double)(SA[0]) };
    double Ap[1];
    MAGMA_D_REAL(Ap[0]) = (double)MAGMA_S_REAL(SA[0]);
#if defined(PRECISION_z) || defined(PRECISION_c)
    MAGMA_D_IMAG(Ap[0]) = (double)MAGMA_S_IMAG(SA[0]);
#endif

    do {
        SA  += LDSA;
        A[0] = Ap[0];
        //Ap[0]= (double)(SA[0]);
        MAGMA_D_REAL(Ap[0]) = (double)MAGMA_S_REAL(SA[0]);
#if defined(PRECISION_z) || defined(PRECISION_c)
        MAGMA_D_IMAG(Ap[0]) = (double)MAGMA_S_IMAG(SA[0]);
#endif
        A   += LDA;

    } while (SA < SAend);

    A[0] = Ap[0];
}

__kernel void 
slag2d_special(int M, int N, 
               __global float *SA, int SA_offset, int LDSA, 
               __global double *A, int A_offset,  int LDA ) 
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
    //double Ap[1] = { (double)(SA[0]) };
    double Ap[1];
    MAGMA_D_REAL(Ap[0]) = (double)MAGMA_S_REAL(SA[0]);
#if defined(PRECISION_z) || defined(PRECISION_c)
    MAGMA_D_IMAG(Ap[0]) = (double)MAGMA_S_IMAG(SA[0]);
#endif
    A[0] = Ap[0];
}


