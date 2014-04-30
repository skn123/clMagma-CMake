/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zgesv.cpp normal z -> d, Fri Jan 10 15:51:18 2014

*/
#include "common_magma.h"

extern "C" magma_err_t
magma_dgesv(     magma_int_t n, magma_int_t nrhs,
                 double *A, magma_int_t lda,
                 magma_int_t *ipiv,
                 double *B, magma_int_t ldb,
                 magma_int_t *info,
                 magma_queue_t *queue )
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose
    =======
    Solves a system of linear equations
       A * X = B
    where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
    The LU decomposition with partial pivoting and row interchanges is
    used to factor A as
       A = P * L * U,
    where P is a permutation matrix, L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.

    Arguments
    =========
    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input/output) DOUBLE_PRECISION array, dimension (LDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) DOUBLE_PRECISION array, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */

    magma_int_t num_gpus, ldda, lddb;
    
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    } else if (ldb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }
    
    /* If single-GPU and allocation suceeds, use GPU interface. */
    num_gpus = magma_num_gpus();
    magmaDouble_ptr dA, dB;
    if ( num_gpus > 1 ) {
        goto CPU_INTERFACE;
    }
    ldda = ((n+31)/32)*32;
    lddb = ldda;
    if ( MAGMA_SUCCESS != magma_dmalloc( &dA, ldda*n )) {
        goto CPU_INTERFACE;
    }
    if ( MAGMA_SUCCESS != magma_dmalloc( &dB, lddb*nrhs )) {
        magma_free( dA );
        goto CPU_INTERFACE;
    }
    magma_dsetmatrix( n, n, A, 0, lda, dA, 0, ldda, queue[0] );
    magma_dgetrf2_gpu( n, n, dA, 0, ldda, ipiv, info, queue );
    if ( *info == MAGMA_ERR_DEVICE_ALLOC ) {
        magma_free( dA );
        magma_free( dB );
        goto CPU_INTERFACE;
    }
    magma_dgetmatrix( n, n, dA, 0, ldda, A, 0, lda, queue[0] );
    if ( *info == 0 ) {
        magma_dsetmatrix( n, nrhs, B, 0, ldb, dB, 0, lddb, queue[0] );
        magma_dgetrs_gpu( MagmaNoTrans, n, nrhs, dA, 0, ldda, ipiv, dB, 0, lddb, info, queue[0]);
        magma_dgetmatrix( n, nrhs, dB, 0, lddb, B, 0, ldb, queue[0]);
    }
    magma_free( dA );
    magma_free( dB );
    return *info;

CPU_INTERFACE:
    /* If multi-GPU or allocation failed, use CPU interface and LAPACK.
     * Faster to use LAPACK for getrs than to copy A to GPU. */
    magma_dgetrf( n, n, A, lda, ipiv, info, queue );
    if ( *info == 0 ) {
       lapackf77_dgetrs( lapack_const(MagmaNoTrans), &n, &nrhs, 
                         A, &lda, ipiv, B, &ldb, info );
    }
    return *info;
}
