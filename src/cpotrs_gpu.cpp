/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014
                                                                                                              
       @generated from zpotrs_gpu.cpp normal z -> c, Fri Jan 10 15:51:17 2014
*/

#include <stdio.h>
#include "common_magma.h"


extern "C" magma_err_t
magma_cpotrs_gpu(magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                 magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
                 magmaFloatComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
                 magma_err_t *info, magma_queue_t queue )
{
/*  -- clMagma (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014
 
    Purpose
    =======

    CPOTRS solves a system of linear equations A*X = B with a Hermitian
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by CPOTRF.

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

    dA      (input) COMPLEX array on the GPU, dimension (LDDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**H*U or A = L*L**H, as computed by CPOTRF.

    LDDA    (input) INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    dB      (input/output) COMPLEX array on the GPU, dimension (LDDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDDB    (input) INTEGER
            The leading dimension of the array B.  LDDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */

    magmaFloatComplex z_one = MAGMA_C_MAKE(  1.0, 0.0 );
    
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

    if( uplo== MagmaUpper){
        if ( nrhs == 1) {
            chk(magma_ctrsv(MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue));
            chk(magma_ctrsv(MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue));
        } else {
            chk(magma_ctrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
            chk(magma_ctrsm(MagmaLeft, MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
        }
    }
    else{
        if ( nrhs == 1) {
            chk(magma_ctrsv(MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue ));
            chk(magma_ctrsv(MagmaLower, MagmaConjTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue ));
        } else {
            chk(magma_ctrsm(MagmaLeft, MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
            chk(magma_ctrsm(MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
        }
    }
    chk( magma_queue_sync( queue ));
    return *info;
}
