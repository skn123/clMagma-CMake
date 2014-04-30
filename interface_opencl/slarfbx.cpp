/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zlarfbx.cpp normal z -> s, Fri Jan 10 15:51:19 2014

*/

#include <stdio.h>

#include "magmablas.h"
#include "CL_MAGMA_RT.h"
#include "common_magma.h"

//#define BLOCK_SIZE 768
#define BLOCK_SIZE 256

/*
    Apply a real block reflector H to a real vector C from the left
    (i.e., C = H C). H is represented in the form
          H = I - V T V'
    where T is the real k-by-k upper triangular matrix in the 
    representation of the block reflector, and V is a real block of
    k elementary reflectors. 
*/
extern "C" magma_err_t
magma_slarfbx_gpu(int m, int k, magmaFloat_ptr V, size_t V_offset, int ldv,
                  magmaFloat_ptr T, size_t T_offset, int ldt, 
                  magmaFloat_ptr c, size_t c_offset, 
                  magmaFloat_ptr dwork, size_t dwork_offset, 
                  magma_queue_t queue)
{
    cl_int ciErrNum;                // Error code var
    cl_kernel ckKernel=NULL;

    ckKernel = rt->KernelPool["magma_sgemv_kernel1"];   // in slarfbx.cl
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    int nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&m );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&V );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&V_offset );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&ldv   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&c );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&c_offset    );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&dwork );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&dwork_offset    );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    size_t GlobalWorkSize[1]={0}, LocalWorkSize[1]={0};
    
    LocalWorkSize[0] = BLOCK_SIZE;
    
    GlobalWorkSize[0] = k*LocalWorkSize[0];
    
    /* dwork = V' c                   */
    //magma_sgemv_kernel1<<< k, BLOCK_SIZE, 0, magma_stream >>>(m, V, ldv, c, dwork); 
    
    // launch kernel magma_sgemv_kernel1
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }

    //clFlush(queue);
    clFinish(queue);
    
    ckKernel = rt->KernelPool["magma_strmv_tkernel"];   // in slarfx.cl
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&T );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&T_offset );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&ldt );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&dwork   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&dwork_offset );
    magmaFloat_ptr dwork2 = dwork;
    size_t dwork2_offset = dwork_offset + k;
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&dwork2 );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&dwork2_offset    );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    LocalWorkSize[0] = k;
    GlobalWorkSize[0] = k*LocalWorkSize[0];
    
    /* dwork = T' dwork               */
    //magma_strmv_tkernel<<< k, k, 0, magma_stream >>>( T, ldt, dwork, dwork+k);
    
    // launch kernel magma_strmv_tkernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }

    //clFlush(queue);
    clFinish(queue);
    
    LocalWorkSize[0] = BLOCK_SIZE;
    GlobalWorkSize[0] = (m+BLOCK_SIZE-1)/BLOCK_SIZE*LocalWorkSize[0];

    ckKernel = rt->KernelPool["magma_sgemv_kernel2"];   // in slarfbx.cl
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&m );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&k );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&V );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&V_offset   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&ldv );
    dwork2 = dwork;
    dwork2_offset = dwork_offset + k;
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&dwork2 );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&dwork2_offset    );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&c );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&c_offset    );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    // launch kernel magma_sgemv_kernel2
    /* c = c - V dwork                */
    //magma_sgemv_kernel2<<< blocks3, threads3, 0, magma_stream >>>( m, k, V, ldv, dwork+k, c);
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
