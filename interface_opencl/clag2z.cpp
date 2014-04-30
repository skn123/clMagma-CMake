/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @precisions mixed zc -> ds
*/

#define PRECISION_z


#include <stdio.h>

#include "magmablas.h"
#include "CL_MAGMA_RT.h"
#include "common_magma.h"

extern "C" magma_err_t 
magmablas_clag2z_64_64_16_4_v2( magma_int_t M, magma_int_t N, 
                                magmaFloatComplex_ptr SA, size_t SA_offset, magma_int_t LDSA, 
                                magmaDoubleComplex_ptr A, size_t A_offset, magma_int_t LDA,
                                magma_queue_t queue)
{
    if( M == 0 || N==0 ) {
        printf("One of the dimension is ZERO\n");
        exit(-1);
    }
    
    cl_int ciErrNum;                // Error code var
    cl_kernel ckKernel=NULL;

    // in clag2z.cl
    if(N>1){
        ckKernel = rt->KernelPool["clag2z_generic"];
    }else{
        ckKernel = rt->KernelPool["clag2z_special"];
    }
    if (!ckKernel)
    {
        printf ("Error: cannot locate kernel in line %d, file %s\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    //dim3 threads( 16, 4 );
    //dim3 grid(M/64+(M%64!=0),1);
     
    size_t GlobalWorkSize[2]={0,0}, LocalWorkSize[2]={0,0};
    LocalWorkSize[0] = 16;
    LocalWorkSize[1] = 4;

    GlobalWorkSize[0] = (M/64+(M%64!=0))*LocalWorkSize[0];
    GlobalWorkSize[1] = 1*LocalWorkSize[1];
    
    int nn = 0;
    ciErrNum  = clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&M   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&N );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaFloatComplex_ptr), (void*)&SA     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&SA_offset   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&LDSA   );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(magmaDoubleComplex_ptr), (void*)&A );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&A_offset     );
    ciErrNum |= clSetKernelArg( ckKernel, nn++, sizeof(int), (void*)&LDA     );
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clSetKernelArg at %d in file %s!\n", __LINE__, __FILE__);
        return MAGMA_ERR_UNKNOWN;
    }
    
    ciErrNum = clEnqueueNDRangeKernel(
        queue, ckKernel, 2, NULL, GlobalWorkSize, LocalWorkSize, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueNDRangeKernel at %d in file %s \"%s\"\n",
            __LINE__, __FILE__, rt->GetErrorCode(ciErrNum));
        return MAGMA_ERR_UNKNOWN;
    }
    
    clFlush(queue);
    return MAGMA_SUCCESS;
}


extern "C" magma_err_t
magmablas_clag2z( magma_int_t m, magma_int_t n , 
                  magmaDoubleComplex_ptr SA, size_t SA_offset,
                  magma_int_t ldsa, 
                  magmaFloatComplex_ptr A, size_t A_offset, 
                  magma_int_t lda, 
                  magma_int_t *info, magma_queue_t queue ) 
{
/*
    Purpose
    =======
    
    CLAG2Z converts a SINGLE PRECISION matrix, SA, to a DOUBLE
    PRECISION matrix, A.
    
    Note that while it is possible to overflow while converting
    from double to single, it is not possible to overflow when
    converting from single to double.
        
    Arguments
    =========
    
    M       (input) INTEGER
            The number of lines of the matrix A.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    SA      (input) REAL array, dimension (LDSA,N)
            On entry, the M-by-N coefficient matrix SA.
    
    LDSA    (input) INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).
    
    A       (output) DOUBLE PRECISION array, dimension (LDA,N)
            On exit, the M-by-N coefficient matrix A.
    
    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */
     

    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,m) )
        *info = -4;
    else if ( ldsa < max(1,m) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        //return *info;
    }
 
    return magmablas_clag2z_64_64_16_4_v2( m, n, SA, SA_offset, ldsa, A, A_offset, lda, queue);
}
