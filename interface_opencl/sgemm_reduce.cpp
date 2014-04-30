/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zgemm_reduce.cpp normal z -> s, Fri Jan 10 15:51:19 2014

*/

#include "common_magma.h"
#include <stdio.h>
#include "CL_MAGMA_RT.h"

//#define NUM_THREADS 1024
#define NUM_THREADS 256

///////////////////////////////////////////////////////////////////////////////////////////////////
// size of work for a thread block
#define BLK_M 8
#define BLK_N 8

#define BLK_K (NUM_THREADS / (BLK_M * BLK_N))

extern "C" magma_err_t
magmablas_sgemm_reduce(magma_int_t m, magma_int_t n, magma_int_t k,
                       float alpha, magmaFloat_ptr d_A, size_t d_A_offset, magma_int_t lda,
                       magmaFloat_ptr d_B, size_t d_B_offset, magma_int_t ldb,
                       float beta,        magmaFloat_ptr d_C, size_t d_C_offset, magma_int_t ldc,
                       magma_queue_t queue)
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

   Purpose
   =======
   SGEMM_REDUCE  performs one of the matrix-matrix operations

      C := alpha* A' B  + beta*C,

   where alpha and beta are scalars, and A, B and C are matrices, with A
   an k-by-m matrix, B a k-by-n matrix, and C an m-by-n matrix. 

   This routine is tuned for m, n << k. Typically, m and n are expected
   less than 128. 
   =====================================================================    */

    if (m%BLK_M!=0 || n%BLK_N!=0) {
        printf("sgemm_reduce works only for m and n divisible by \n");
        printf("correspondingly %d and %d. Calling magma_sgemm ...\n.", 
               BLK_M, BLK_N);
        magma_sgemm( MagmaTrans, MagmaNoTrans,
                     m, n, k, 
                     alpha, d_A, d_A_offset, lda, d_B, d_B_offset, ldb, beta, d_C, d_C_offset, ldc,
                     queue );
    }   
    else {
        cl_int ciErrNum;                // Error code var
        cl_kernel ckKernel=NULL;
    
        ckKernel = rt->KernelPool["magmablas_sgemm_reduce_kernel"];
        if (!ckKernel)
        {
            printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
            return MAGMA_ERR_UNKNOWN;
        }
    
        int nn = 0;
        ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&k   );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(float), (void*)&alpha );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&d_A     );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&d_A_offset   );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&lda );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&d_B     );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&d_B_offset   );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&ldb );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(float), (void*)&beta     );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&d_C     );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&d_C_offset   );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&ldc );
        if (ciErrNum != CL_SUCCESS)
        {
            printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
            return MAGMA_ERR_UNKNOWN;
        }
   
        /*
        dim3  blocks( m/BLK_M, n/BLK_N );
        dim3 threads( BLK_K, BLK_M, BLK_N );
        */
        size_t GlobalWorkSize[3]={0,0,0}, LocalWorkSize[3]={0,0,0};
    
        LocalWorkSize[0] = BLK_K;
        LocalWorkSize[1] = BLK_M;
        LocalWorkSize[2] = BLK_N;
    
        GlobalWorkSize[0] = (m/BLK_M)*LocalWorkSize[0];
        GlobalWorkSize[1] = (n/BLK_N)*LocalWorkSize[1];
        GlobalWorkSize[2] = 1*LocalWorkSize[2];
    
        // launch kernel
        ciErrNum = clEnqueueNDRangeKernel(
            queue, ckKernel, 3, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS)
        {
            printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
                __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
            return MAGMA_ERR_UNKNOWN;
        }
        clFlush(queue); 
        return MAGMA_SUCCESS;
    }
}

