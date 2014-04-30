/*
     -- clMAGMA (version 1.1.0) --
        Univ. of Tennessee, Knoxville
        Univ. of California, Berkeley
        Univ. of Colorado, Denver
        @date January 2014
  
        @generated from zpotrf_gpu.cpp normal z -> d, Fri Jan 10 15:51:17 2014

*/

#include <stdio.h>
#include "common_magma.h"

magma_err_t
magma_dpotrf_gpu( magma_uplo_t   uplo, magma_int_t    n,
          magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
          magma_err_t*   info, magma_queue_t queue )
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose
    =======
    DPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix dA.

    The factorization has the form
       dA = U\*\*H * U,  if UPLO = 'U', or
       dA = L  * L\*\*H,  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    =========
    UPLO    (input) INTEGER
            = MagmaUpper:  Upper triangle of dA is stored;
            = MagmaLower:  Lower triangle of dA is stored.

    N       (input) INTEGER
            The order of the matrix dA.  N >= 0.

    dA      (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix dA.  If UPLO = 'U', the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U\*\*H*U or dA = L*L\*\*H.

    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            dividable by 16.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.
    =====================================================================   */

// produces pointer and offset as two args to magmaBLAS routines
#define dA(i,j)  dA, ( (dA_offset) + (i) + (j)*ldda )

// produces pointer as single arg to BLAS routines
#define A(i,j)  &A[ (i) + (j)*lda ]

    magma_int_t j, jb, nb;
    double  z_one = MAGMA_D_MAKE(  1.0, 0.0 );
    double mz_one = MAGMA_D_MAKE( -1.0, 0.0 );
    double    one =  1.0;
    double  m_one = -1.0;
    double* work;
    magma_err_t err;
    
    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        *info = -1;
    } else if ( n < 0 ) {
        *info = -2;
    } else if ( ldda < max(1,n) ) {
        *info = -4;
    }
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    nb = magma_get_dpotrf_nb( n );
    
    err = magma_dmalloc_cpu(  &work, nb*nb );
    if ( err != MAGMA_SUCCESS ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    magma_event_t event = NULL;
    
    if ((nb <= 1) || (nb >= n)) {
        // use unblocked code
        chk( magma_dgetmatrix( n, n, dA, dA_offset, ldda, work, 0, n, queue ));
        lapackf77_dpotrf( lapack_uplo_const(uplo), &n, work, &n, info );
        chk( magma_dsetmatrix( n, n, work, 0, n, dA, dA_offset, ldda, queue ));
    }
    else {
        if ( uplo == MagmaUpper ) {
            // --------------------
            // compute Cholesky factorization A = U'*U
            // using the left looking algorithm
            for( j = 0; j < n; j += nb ) {
                // apply all previous updates to diagonal block
                jb = min( nb, n-j );
                if ( j > 0 ) {
                    chk( magma_dsyrk( MagmaUpper, MagmaTrans, jb, j,
                                      m_one, dA(0,j), ldda,
                                      one, dA(j,j), ldda, queue ));
                }
                
                // start asynchronous data transfer
                chk( magma_dgetmatrix_async( jb, jb, dA(j,j), ldda, work, 0, jb, queue, &event ));

                // apply all previous updates to block row right of diagonal block
                if ( j+jb < n ) {
                    chk( magma_dgemm( MagmaTrans, MagmaNoTrans,
                        jb, n-j-jb, j,
                        mz_one, dA(0, j   ), ldda,
                                dA(0, j+jb), ldda,
                        z_one,  dA(j, j+jb), ldda, queue ));
                }
                
                // simultaneous with above dgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                chk( magma_event_sync( event ));
                lapackf77_dpotrf( MagmaUpperStr, &jb, work, &jb, info );
                if ( *info != 0 ) {
                    assert( *info > 0 );
                    *info += j;
                    break;
                }
                chk( magma_dsetmatrix_async( jb, jb, work, 0, jb, dA(j,j), ldda, queue, &event ));
                
                // apply diagonal block to block row right of diagonal block
                if ( j+jb < n ) {
                    chk( magma_event_sync( event ));
                    chk( magma_dtrsm(
                        MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                        jb, n-j-jb,
                        z_one, dA(j, j),    ldda,
                               dA(j, j+jb), ldda, queue ));
                }
            }
        }
        else {
            // --------------------
            // compute Cholesky factorization A = L*L'
            // using the left looking algorithm
            for( j = 0; j < n; j += nb ) {
                // apply all previous updates to diagonal block
                jb = min( nb, n-j );
                if ( j>0 ) {
                    chk( magma_dsyrk( MagmaLower, MagmaNoTrans, jb, j,
                                      m_one, dA(j, 0), ldda,
                                      one, dA(j, j), ldda, queue ));
                }

                // start asynchronous data transfer
                chk( magma_dgetmatrix_async( jb, jb, dA(j,j), ldda, work, 0, jb, queue, &event ));

                // apply all previous updates to block column below diagonal block
                if ( j+jb < n ) {
                    chk( magma_dgemm( MagmaNoTrans, MagmaTrans,
                        n-j-jb, jb, j,
                        mz_one, dA(j+jb, 0), ldda,
                                dA(j,    0), ldda,
                        z_one,  dA(j+jb, j), ldda, queue ));
                }
                
                // simultaneous with above dgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                chk( magma_event_sync( event ));
                lapackf77_dpotrf( MagmaLowerStr, &jb, work, &jb, info );
                if ( *info != 0 ) {
                    assert( *info > 0 );
                    *info += j;
                    break;
                }
                chk( magma_dsetmatrix_async( jb, jb, work, 0, jb, dA(j,j), ldda, queue, &event ));
                
                // apply diagonal block to block column below diagonal
                if ( j+jb < n ) {
                    chk( magma_event_sync( event ));
                    chk( magma_dtrsm(
                        MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                        n-j-jb, jb,
                        z_one, dA(j, j   ), ldda,
                               dA(j+jb, j), ldda, queue ));
                }
            }
        }
    }
    
    chk( magma_queue_sync( queue ));
    chk( magma_free_cpu( work ));
    
    return *info;
}
