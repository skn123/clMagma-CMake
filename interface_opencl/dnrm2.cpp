/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from dznrm2.cpp normal z -> d, Fri Jan 10 15:51:19 2014

*/
#include "common_magma.h"
#include <stdio.h>
#include "CL_MAGMA_RT.h"

//#define BLOCK_SIZE  512
#define BLOCK_SIZE  256
#define BLOCK_SIZEx  32
//#define BLOCK_SIZEy  16
#define BLOCK_SIZEy  8

#define PRECISION_d

/*
    Adjust the norm of c to give the norm of c[k+1:], assumin that
    c was changed with orthogonal transformations.
*/
extern "C" magma_err_t
magmablas_dnrm2_adjust(int k, magmaDouble_ptr xnorm, size_t xnorm_offset, magmaDouble_ptr c, size_t c_offset, magma_queue_t queue)
{
    size_t GlobalWorkSize[1]={0}, LocalWorkSize[1]={0};
    
    LocalWorkSize[0] = k;
    GlobalWorkSize[0] = 1*LocalWorkSize[0];
    
    cl_int ciErrNum;                // Error code var
    cl_kernel ckKernel=NULL;
    
    ckKernel = rt->KernelPool["magmablas_dnrm2_adjust_kernel"];
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    int nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(magmaDouble_ptr), (void*)&xnorm   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&xnorm_offset );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaDouble_ptr), (void*)&c     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&c_offset   );
    //magma_dnrm2_adjust_kernel<<< 1, k, 0, magma_stream >>> (xnorm, c);
    // launch kernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        printf("block: %d,    group: %d\n", (int) LocalWorkSize[0], (int) GlobalWorkSize[0]);
        return MAGMA_ERR_UNKNOWN;
    }

    clFlush(queue);
    return MAGMA_SUCCESS;
}
//==============================================================================

/*
   Compute the dnrm2 of da, da+ldda, ..., da +(num-1)*ldda where the vectors are
   of size m. The resulting norms are written in the dxnorm array. 
   The computation can be done using num blocks (default) or on one SM (commented).
*/
extern "C" magma_err_t
magmablas_dnrm2(int m, int num, magmaDouble_ptr da, size_t da_offset, magma_int_t ldda, 
                 magmaDouble_ptr dxnorm, size_t dxnorm_offset, magma_queue_t queue) 
{
    size_t GlobalWorkSize[1]={0}, LocalWorkSize[1]={0};
   
    LocalWorkSize[0] = BLOCK_SIZE;
    GlobalWorkSize[0] = num * LocalWorkSize[0];
    
    cl_int ciErrNum;                // Error code var
    cl_kernel ckKernel=NULL;
    
    ckKernel = rt->KernelPool["magmablas_dnrm2_kernel"];
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    int nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&m   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaDouble_ptr), (void*)&da );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&da_offset     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&ldda   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaDouble_ptr), (void*)&dxnorm );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&dxnorm_offset     );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    // launch kernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }
    
    clFlush(queue);
    return MAGMA_SUCCESS;
}
