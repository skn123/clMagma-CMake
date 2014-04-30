/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014
                                                                                                              
       @precisions normal z -> s d c
*/

#include <stdio.h>
#include "common_magma.h"


extern "C" magma_err_t
magma_zposv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                 magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
                 magmaDoubleComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
                 magma_err_t *info, magma_queue_t queue )
{
/*  -- clMagma (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014
 
    Purpose
    =======

    ZPOSV computes the solution to a complex system of linear equations
       A * X = B,
    where A is an N-by-N Hermitian positive definite matrix and X and B
    are N-by-NRHS matrices.
    The Cholesky decomposition is used to factor A as
       A = U**H * U,  if UPLO = 'U', or
       A = L * L**H,  if UPLO = 'L',
    where U is an upper triangular matrix and  L is a lower triangular
    matrix.  The factored form of A is then used to solve the system of
    equations A * X = B.

    Arguments
    =========
 
    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the Hermitian matrix dA.  If UPLO = 'U', the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H*U or dA = L*L**H.

    LDDA    (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    dB      (input/output) COMPLEX_16 array on the GPU, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDDB    (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */

    magma_err_t ret;
    
    *info = 0 ;
    if( (uplo != MagmaUpper) && (uplo != MagmaLower) )
        *info = -1;
    if( n < 0 )
        *info = -2;
    if( nrhs < 0)
        *info = -3;
    if ( ldda < max(1, n) )
        *info = -5;
    if ( lddb < max(1, n) )
        *info = -7;
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if ( (n == 0) || (nrhs == 0) ) {
        return *info;
    }

    ret = magma_zpotrf_gpu(uplo, n, dA, 0, ldda, info, queue);
    if ( (ret != MAGMA_SUCCESS) || ( *info != 0 ) ) {
        return ret;
    }

    ret = magma_zpotrs_gpu(uplo, n, nrhs, dA, 0, ldda, dB, 0, lddb, info, queue);
    if ( (ret != MAGMA_SUCCESS) || ( *info != 0 ) ) {
        return ret;
    }

    return *info;
}
