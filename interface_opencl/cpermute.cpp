/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @generated from zpermute.cpp normal z -> c, Fri Jan 10 15:51:19 2014
 */

#include <stdio.h>

#include "magmablas.h"
#include "CL_MAGMA_RT.h"

#define BLOCK_SIZE 32

#define min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int n;
    int lda;
    int j0;
    int npivots;
    short ipiv[BLOCK_SIZE];
} claswp_params_t2;

// ----------------------------------------
void claswp3(
        cl_mem dA, size_t offset, claswp_params_t2 params, magma_queue_t queue )
{
    cl_int ciErrNum;                // Error code var
    cl_kernel ckKernel=NULL;

    ckKernel = rt->KernelPool["myclaswp2"];
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return;
    }

    int nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(cl_mem),           (void*)&dA     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int),           (void*)&offset );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(claswp_params_t2), (void*)&params );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s, %s\n", __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return;
    }

    size_t GlobalWorkSize[2]={0,0}, LocalWorkSize[2]={0,0};

    LocalWorkSize[0] = BLOCK_SIZE;
    LocalWorkSize[1] = 1;

    GlobalWorkSize[0] = ((params.n+BLOCK_SIZE-1) / BLOCK_SIZE)*LocalWorkSize[0];
    GlobalWorkSize[1] = 1;

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


// ----------------------------------------
// offset is not added to ipiv
magma_err_t
magma_cpermute_long2(int n,
        cl_mem dAT, size_t dAT_offset, int lda,
        int *ipiv, int nb, int ind,
        magma_queue_t queue )
{
    int k;

    for( k = 0; k < nb-BLOCK_SIZE; k += BLOCK_SIZE )
    {
        claswp_params_t2 params = { n, lda, ind + k, BLOCK_SIZE };
        for( int j = 0; j < BLOCK_SIZE; j++ )
        {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1;
            ipiv[ind + k + j] += ind;
        }
        claswp3( dAT, dAT_offset, params, queue );
    }

    int num_pivots = nb - k;

    claswp_params_t2 params = { n, lda, ind + k, num_pivots };
    for( int j = 0; j < num_pivots; j++ )
    {
        params.ipiv[j] = ipiv[ind + k + j] - k - 1;
        ipiv[ind + k + j] += ind;
    }
    claswp3( dAT, dAT_offset, params, queue );

    return MAGMA_SUCCESS;
}


// ----------------------------------------
// offset is already added to ipiv, used in cgetrf2_mgpu
magma_err_t
magma_cpermute_long3(int n,
        cl_mem dAT, size_t dAT_offset, int lda,
        int *ipiv, int nb, int ind,
        magma_queue_t queue )
{
    for( int k = 0; k < nb; k += BLOCK_SIZE ) {
        int npivots = min( BLOCK_SIZE, nb-k );
        // fields are:              n  lda  j0       npivots
        claswp_params_t2 params = { n, lda, ind + k, npivots };
        for( int j = 0; j < BLOCK_SIZE; ++j ) {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1 - ind;
        }
        claswp3(dAT, dAT_offset, params, queue );
    }

    return MAGMA_SUCCESS;
}


