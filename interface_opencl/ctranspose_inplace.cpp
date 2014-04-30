/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @generated from ztranspose_inplace.cpp normal z -> c, Fri Jan 10 15:51:19 2014
 */

#include <stdio.h>

#include "magmablas.h"
#include "CL_MAGMA_RT.h"

//#define NB 16
#define CSIZE_2SHARED 16

magma_err_t
magma_ctranspose_inplace(
    cl_mem A, size_t offset, int lda, int m, magma_queue_t queue )
{
    cl_int ciErrNum;                // Error code var
    //int in = m / NB;
    int in = m / CSIZE_2SHARED;
    cl_kernel ckKernel=NULL;
    if (in&1)
    {
        //printf ("running odd kernel\n");
        ckKernel = rt->KernelPool["ctranspose_inplace_odd_kernel"];
    }
    else
    {
        //printf ("running even kernel\n");
        ckKernel = rt->KernelPool["ctranspose_inplace_even_kernel"];
    }
    
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    int nn = 0, half;
    
    if (in&1)
        half = in/2+1;
    else
        half = in/2;
    
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(cl_mem), (void*)&A      );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&offset );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&lda    );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&half   );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    size_t GlobalWorkSize[2]={0,0}, LocalWorkSize[2]={0,0};
    
    //LocalWorkSize[0] = NB;
    //LocalWorkSize[1] = NB/2;
    
    LocalWorkSize[0] = CSIZE_2SHARED;
    LocalWorkSize[1] = CSIZE_2SHARED/2;
    
    if (in&1)
    {
        GlobalWorkSize[0] = (in    )*LocalWorkSize[0];
        GlobalWorkSize[1] = (in/2+1)*LocalWorkSize[1];
    }
    else
    {
        GlobalWorkSize[0] = (in+1)*LocalWorkSize[0];
        GlobalWorkSize[1] = (in/2)*LocalWorkSize[1];
    }
    
    // launch kernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 2, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }
    
    return MAGMA_SUCCESS;
}
