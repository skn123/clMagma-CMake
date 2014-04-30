/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @generated from zauxiliary.cpp normal z -> s, Fri Jan 10 15:51:19 2014
 */

#include <stdio.h>

#include "magmablas.h"
#include "CL_MAGMA_RT.h"

#define slaset_threads 64

void szero_nbxnb_block(int nb, cl_mem dA, size_t dA_offset, int ldda, magma_queue_t queue)
{
    cl_int ciErrNum;
    cl_kernel ckKernel=NULL;

    ckKernel = rt->KernelPool["sset_nbxnb_to_zero"];
    if(!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return;
    }

    int nn = 0;
    int offset = (int)dA_offset;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(cl_int),           (void*)&nb     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_mem),           (void*)&dA );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int),           (void*)&offset );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&ldda );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s, %s\n", __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return;
    }

    size_t LocalWorkSize[1] = {32};
    size_t GlobalWorkSize[1] = {32*LocalWorkSize[0]};

    // launch kernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 1, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return;
    }
}

/*    ///////////////////////////////////////////////////////////////////////////
    -- Set the m x n matrix pointed by A to 0 on the GPU
*/
extern "C" void
magmablas_slaset(int uplo, magma_int_t m, magma_int_t n, cl_mem A, size_t A_offset, magma_int_t lda, magma_queue_t queue)
{
    size_t LocalWorkSize[2] = {slaset_threads, 1};
    size_t GlobalWorkSize[2] = {
        (m/slaset_threads+(m % slaset_threads != 0))*LocalWorkSize[0],
        (n/32+(n%32!=0))*LocalWorkSize[1]
    };
    
    cl_int ciErrNum;
    cl_kernel ckKernel=NULL;
    
    if (m!=0 && n!=0) {
        if (uplo == MagmaLower) {
            ckKernel = rt->KernelPool["slaset_lower"];
            if(!ckKernel){
                printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
                return;
            }
        }
        else if (uplo == MagmaUpper) {
            ckKernel = rt->KernelPool["slaset_upper"];
            if(!ckKernel){
                printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
                return;
            }
        }
        else {
            ckKernel = rt->KernelPool["slaset"];
            if(!ckKernel){
                printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
                return;
            }
        }
    } else {
        return;
    }

    int offset = (int)A_offset;
    int nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(cl_int),           (void*)&m);
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int),           (void*)&n );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_mem),           (void*)&A);
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int),              (void*)&offset );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int),              (void*)&lda );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s, %s\n", __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return;
    }

    // launch kernel
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 2, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return;
    }
}

