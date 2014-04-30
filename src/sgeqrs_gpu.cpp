/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       
       @date January 2014
       
       @generated from zgeqrs_gpu.cpp normal z -> s, Fri Jan 10 15:51:18 2014
*/

#include <stdio.h>
#include "common_magma.h"

extern "C" magma_err_t
magma_sgeqrs_gpu(magma_int_t m, magma_int_t n, magma_int_t nrhs,
                 magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
                 float *tau,   magmaFloat_ptr dT, size_t dT_offset,
                 magmaFloat_ptr dB, size_t dB_offset, magma_int_t lddb,
                 float *hwork, magma_int_t lwork,
                 magma_int_t *info, magma_queue_t queue)
{
/*  -- clMagma (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose
    =======
    Solves the least squares problem
           min || A*X - C ||
    using the QR factorization A = Q*R computed by SGEQRF_GPU.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    NRHS    (input) INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    A       (input) REAL array on the GPU, dimension (LDDA,N)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,n, as returned by
            SGEQRF_GPU in the first n columns of its array argument A.

    LDDA    (input) INTEGER
            The leading dimension of the array A, LDDA >= M.

    TAU     (input) REAL array, dimension (N)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_SGEQRF_GPU.

    DB      (input/output) REAL array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    DT      (input) REAL array that is the output (the 6th argument)
            of magma_sgeqrf_gpu of size
            2*MIN(M, N)*NB + ((N+31)/32*32 )* MAX(NB, NRHS).
            The array starts with a block of size MIN(M,N)*NB that stores
            the triangular T matrices used in the QR factorization,
            followed by MIN(M,N)*NB block storing the diagonal block
            inverses for the R matrix, followed by work space of size
            ((N+31)/32*32 )* MAX(NB, NRHS).

    LDDB    (input) INTEGER
            The leading dimension of the array DB. LDDB >= M.

    HWORK   (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK, LWORK >= max(1,NRHS).
            For optimum performance LWORK >= (M-N+NB)*(NRHS + 2*NB), where
            NB is the blocksize given by magma_get_sgeqrf_nb( M ).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the WORK array.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */

   #define a_ref(a_1,a_2)  dA, (dA_offset + (a_1) + (a_2)*(ldda))
   #define d_ref(a_1)      dT, (dT_offset + (lddwork+(a_1))*nb)

    float c_zero    = MAGMA_S_ZERO;
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;
    magmaFloat_ptr dwork;
    magma_int_t i, k, lddwork, rows, ib;
    magma_int_t ione = 1;

    magma_int_t nb     = magma_get_sgeqrf_nb(m);
    magma_int_t lwkopt = (m-n+nb)*(nrhs+2*nb);
    long int lquery = (lwork == -1);

    hwork[0] = MAGMA_S_MAKE( (float)lwkopt, 0. );

    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0 || m < n)
        *info = -2;
    else if (nrhs < 0)
        *info = -3;
    else if (ldda < max(1,m))
        *info = -5;
    else if (lddb < max(1,m))
        *info = -8;
    else if (lwork < lwkopt && ! lquery)
        *info = -10;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    k = min(m,n);
    if (k == 0) {
        hwork[0] = c_one;
        return *info;
    }

    /* B := Q' * B */
    magma_sormqr_gpu( MagmaLeft, MagmaTrans,
                      m, nrhs, n,
                      a_ref(0,0), ldda, tau,
                      dB, dB_offset, lddb, hwork, lwork, dT, dT_offset, nb, info, queue );
    if ( *info != 0 ) {
        return *info;
    }

    /* Solve R*X = B(1:n,:) */
    lddwork= k;

    int ldtwork;
    size_t dwork_offset = 0;
    if (nb < k)
      {
        dwork = dT;
        dwork_offset = dT_offset+2*lddwork*nb;
      }
    else
      {
        ldtwork = ( 2*k + ((n+31)/32)*32 )*nb;
        magma_smalloc( &dwork, ldtwork );
      }
    // To do: Why did we have this line originally; seems to be a bug (Stan)?
    //dwork = dT;

    i    = (k-1)/nb * nb;
    ib   = n-i;
    rows = m-i;

    if ( nrhs == 1 ) {
        blasf77_strsv( MagmaUpperStr, MagmaNoTransStr, MagmaNonUnitStr,
                       &ib, hwork,         &rows,
                            hwork+rows*ib, &ione);
    } else {
        blasf77_strsm( MagmaLeftStr, MagmaUpperStr, MagmaNoTransStr, MagmaNonUnitStr,
                       &ib, &nrhs,
                       &c_one, hwork,         &rows,
                               hwork+rows*ib, &rows);
    }
      
    // update the solution vector
    magma_ssetmatrix( ib, nrhs, hwork+rows*ib, 0, rows, dwork, dwork_offset+i, lddwork, queue );

    // update c
    if (nrhs == 1)
        magma_sgemv( MagmaNoTrans, i, ib,
                     c_neg_one, a_ref(0, i), ldda,
                                         dwork, dwork_offset+i, 1,
                     c_one,     dB, dB_offset, 1, queue );
    else
        magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                     i, nrhs, ib,
                     c_neg_one, a_ref(0, i), ldda,
                                dwork, dwork_offset + i,   lddwork,
                     c_one,     dB, dB_offset, lddb, queue );

    int start = i-nb;
    if (nb < k) {
        for (i = start; i >=0; i -= nb) {
            ib = min(k-i, nb);
            rows = m -i;

            if (i + ib < n) {
                if (nrhs == 1) {
                    magma_sgemv( MagmaNoTrans, ib, ib,
                                 c_one,  d_ref(i), ib,
                                 dB, dB_offset+i,      1,
                                 c_zero, dwork, dwork_offset+i,  1, queue );
                    magma_sgemv( MagmaNoTrans, i, ib,
                                 c_neg_one, a_ref(0, i), ldda,
                                 dwork, dwork_offset+i,   1,
                                 c_one,     dB, dB_offset, 1, queue );
                } else {
                    magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                                 ib, nrhs, ib,
                                 c_one,  d_ref(i), ib,
                                 dB, dB_offset+i, lddb,
                                 c_zero, dwork, dwork_offset+i,  lddwork, queue );
                    magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                                 i, nrhs, ib,
                                 c_neg_one, a_ref(0, i), ldda,
                                 dwork, dwork_offset+i, lddwork,
                                 c_one,     dB, dB_offset, lddb, queue );
                }
            }
        }
    }

    magma_scopymatrix( (n), nrhs,
                       dwork, dwork_offset, lddwork,
                       dB, dB_offset,   lddb, queue );

    if (nb >= k)
      magma_free(dwork);

    magma_queue_sync( queue );

    return *info;
}

#undef a_ref
#undef d_ref
