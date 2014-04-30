/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zlag2c.cpp mixed zc -> ds, Fri Jan 10 15:51:19 2014

*/

#define PRECISION_d


#include <stdio.h>

#include "magmablas.h"
#include "CL_MAGMA_RT.h"
#include "common_magma.h"

#define blksize 64

extern "C" magma_err_t 
magmablas_dlag2s( magma_int_t M, magma_int_t N , 
                  magmaDouble_ptr A, size_t A_offset,
                  magma_int_t lda, 
                  magmaFloat_ptr SA, size_t SA_offset, 
                  magma_int_t ldsa, 
                  magma_int_t *info, magma_queue_t queue ) 
{    
/*
    Note
    ====
          - We have to provide INFO at the end that dlag2s isn't doable now. 
          - Transfer a single value TO/FROM CPU/GPU
          - SLAMCH that's needed is called from underlying BLAS
          - Only used in iterative refinement
          - Do we want to provide this in the release?
    
    Purpose
    =======
    DLAG2S converts a DOUBLE PRECISION matrix A to a SINGLE PRECISION
    matrix SA.
    
    RMAX is the overflow for the SINGLE PRECISION arithmetic.
    DLAG2S checks that all the entries of A are between -RMAX and
    RMAX. If not the convertion is aborted and a flag is raised.
        
    Arguments
    =========
    M       (input) INTEGER
            The number of lines of the matrix A.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    A       (input) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N coefficient matrix A.
    
    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    SA      (output) SINGLE PRECISION array, dimension (LDSA,N)
            On exit, if INFO=0, the M-by-N coefficient matrix SA; if
            INFO>0, the content of SA is unspecified.
    
    LDSA    (input) INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).
    
    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value
            = 1:  an entry of the matrix A is greater than the SINGLE PRECISION
                  overflow threshold, in this case, the content
                  of SA in exit is unspecified.
    =====================================================================    */

    *info = 0;
    if ( M < 0 )
        *info = -1;
    else if ( N < 0 )
        *info = -2;
    else if ( lda < max(1,M) )
        *info = -4;
    else if ( ldsa < max(1,M) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        //return *info;
    }
    
    cl_int ciErrNum;                // Error code var
    cl_kernel ckKernel=NULL;

    ckKernel = rt->KernelPool["magmaint_dlag2s"];   // in dlag2s.cl
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    double RMAX = (double)lapackf77_slamch("O");

    //dim3 threads( blksize, 1, 1 );
    //dim3 grid( (M+blksize-1)/blksize, 1, 1);
    size_t GlobalWorkSize[3]={0,0,0}, LocalWorkSize[3]={0,0,0};
    LocalWorkSize[0] = blksize;
    LocalWorkSize[1] = 1;
    LocalWorkSize[2] = 1;

    GlobalWorkSize[0] = ((M+blksize-1)/blksize)*LocalWorkSize[0];
    GlobalWorkSize[1] = 1*LocalWorkSize[1];
    GlobalWorkSize[2] = 1*LocalWorkSize[2];

    // flag is not used in opencl version
    //cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    
    //magmaint_dlag2s<<< grid, threads, 0, magma_stream >>>( M, N, A, lda, SA, ldsa, RMAX ) ; 
    int nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&M   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&N );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaDouble_ptr), (void*)&A     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&A_offset   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&lda   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloat_ptr), (void*)&SA );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&SA_offset     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&ldsa     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(double), (void*)&RMAX   );
   
    // launch kernel magmaint_dlag2s
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 3, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }
    
    //flag is not used in opencl version
    //cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag
    
    clFlush(queue);
    return MAGMA_SUCCESS;
}

