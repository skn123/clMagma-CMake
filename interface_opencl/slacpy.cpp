/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zlacpy.cpp normal z -> s, Fri Jan 10 15:51:19 2014

*/

#include <stdio.h>
#include "common_magma.h"
#include "magmablas.h"
#include "CL_MAGMA_RT.h"

extern "C" void
magmablas_slacpy( magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                  magmaFloat_ptr dA, size_t dA_offset, magma_int_t lda,
                  magmaFloat_ptr dB, size_t dB_offset, magma_int_t ldb,
                  magma_queue_t queue)
{
/*
    Note
  ========
  - UPLO Parameter is disabled
  - Do we want to provide a generic function to the user with all the options?

  Purpose
  =======

  SLACPY copies all or part of a two-dimensional matrix A to another
  matrix B.

  Arguments
  =========

  UPLO    (input) INTEGER
          Specifies the part of the matrix A to be copied to B.
          = 'U':      Upper triangular part
          = 'L':      Lower triangular part
          Otherwise:  All of the matrix A

  M       (input) INTEGER
          The number of rows of the matrix A.  M >= 0.

  N       (input) INTEGER
          The number of columns of the matrix A.  N >= 0.

  A       (input) COMPLEX REAL array, dimension (LDA,N)
          The m by n matrix A.  If UPLO = 'U', only the upper triangle
          or trapezoid is accessed; if UPLO = 'L', only the lower
          triangle or trapezoid is accessed.

  LDA     (input) INTEGER
          The leading dimension of the array A.  LDA >= max(1,M).

  B       (output) COMPLEX REAL array, dimension (LDB,N)
          On exit, B = A in the locations specified by UPLO.

  LDB     (input) INTEGER
          The leading dimension of the array B.  LDB >= max(1,M).

  =====================================================================   */

    size_t LocalWorkSize[1] = {64};
    size_t GlobalWorkSize[1] = {(m/64+(m%64 != 0))*64};
    
    if ( m == 0 || n == 0 )
        return;
    
    if ( uplo == MagmaUpper ) {
        fprintf(stderr, "lacpy upper is not implemented\n");
    }
    else if ( uplo == MagmaLower ) {
        fprintf(stderr, "lacpy lower is not implemented\n");
    }
    else {
        cl_int ciErrNum;
        cl_kernel ckKernel = NULL;
        ckKernel = rt->KernelPool["slacpy_kernel"];
        if(!ckKernel){
            printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
            return;
        }
        
        int offset_A = (int)dA_offset;
        int offset_B = (int)dB_offset;
        int nn = 0;
        ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&m        );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&n        );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_mem), (void*)&dA       );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&offset_A );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&lda      );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_mem), (void*)&dB       );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&offset_B );
        ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(cl_int), (void*)&ldb      );
        if (ciErrNum != CL_SUCCESS){
            printf("Error: clSetKernelArg at %d in file %s, %s\n", __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
            return;
        }

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
}
