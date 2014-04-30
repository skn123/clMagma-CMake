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
magma_zpotrs_gpu(magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
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

    ZPOTRS solves a system of linear equations A*X = B with a Hermitian
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by ZPOTRF.

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

    dA      (input) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**H*U or A = L*L**H, as computed by ZPOTRF.

    LDDA    (input) INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    dB      (input/output) COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDDB    (input) INTEGER
            The leading dimension of the array B.  LDDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */

    magmaDoubleComplex z_one = MAGMA_Z_MAKE(  1.0, 0.0 );
    
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
            chk(magma_ztrsv(MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue));
            chk(magma_ztrsv(MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue));
        } else {
            chk(magma_ztrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
            chk(magma_ztrsm(MagmaLeft, MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
        }
    }
    else{
        if ( nrhs == 1) {
            chk(magma_ztrsv(MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue ));
            chk(magma_ztrsv(MagmaLower, MagmaConjTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue ));
        } else {
            chk(magma_ztrsm(MagmaLeft, MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
            chk(magma_ztrsm(MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
        }
    }
    chk( magma_queue_sync( queue ));
    return *info;
}
