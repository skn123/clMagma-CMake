/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @author Stan Tomov
       @generated from zgeqrf.cpp normal z -> c, Fri Jan 10 15:51:18 2014

*/
#include "common_magma.h"

extern "C" magma_err_t
magma_cgeqrf(magma_int_t m, magma_int_t n,
             magmaFloatComplex *A,    magma_int_t lda, magmaFloatComplex *tau,
             magmaFloatComplex *work, magma_int_t lwork,
             magma_int_t *info,
             magma_queue_t* queue )
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose
    =======
    CGEQRF computes a QR factorization of a COMPLEX M-by-N matrix A:
    A = Q * R. This version does not require work space on the GPU
    passed as input. GPU memory is allocated in the routine.

    If the current stream is NULL, this version replaces it with user defined
    stream to overlap computation with communication.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= max( N*NB, 2*NB*NB ),
            where NB can be obtained through magma_get_cgeqrf_nb(M).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ===============
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).
    =====================================================================    */

    #define  A(i,j) ( A + (i) + (j)*lda )
    #define dA(i,j) dA, dA_offset + (i) + (j)*ldda

    magmaFloatComplex_ptr dA, dwork, dT;
    size_t dA_offset, dwork_offset, dT_offset;

    magmaFloatComplex c_one = MAGMA_C_ONE;

    magma_int_t i, k, lddwork, old_i, old_ib;
    magma_int_t ib, ldda;

    *info = 0;
    magma_int_t nb = magma_get_cgeqrf_nb(min(m, n));

    // need 2*nb*nb to store T and upper triangle of V simultaneously
    magma_int_t lwkopt = max(n*nb, 2*nb*nb);
    work[0] = MAGMA_C_MAKE( (float)lwkopt, 0 );
    int lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    } else if (lwork < max(1, lwkopt) && ! lquery) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    k = min(m,n);
    if (k == 0) {
        work[0] = c_one;
        return *info;
    }

    // largest N for larfb is n-nb (trailing matrix lacks 1st panel)
    lddwork = ((n+31)/32)*32 - nb;
    ldda    = ((m+31)/32)*32;

    magma_int_t num_gpus = magma_num_gpus();
    if( num_gpus > 1 ) {
        /* call multiple-GPU interface  */
        printf("multiple-GPU verison not implemented\n");
        return MAGMA_ERR_NOT_IMPLEMENTED;
        //return magma_cgeqrf4(num_gpus, m, n, A, lda, tau, work, lwork, info);
    }

    // allocate space for dA, dwork, and dT
    if (MAGMA_SUCCESS != magma_cmalloc( &dA, (n*ldda + nb*lddwork + nb*nb) )) {
        /* Switch to the "out-of-core" (out of GPU-memory) version */
        printf("non-GPU-resident version not implemented\n");
        return MAGMA_ERR_NOT_IMPLEMENTED;
        //return magma_cgeqrf_ooc(m, n, A, lda, tau, work, lwork, info);
    }

    dA_offset = 0;

    dwork = dA;
    dwork_offset = n*ldda;

    dT    = dA;
    dT_offset = n*ldda + nb*lddwork;

    if ( (nb > 1) && (nb < k) ) {
        /* Use blocked code initially.
           Asynchronously send the matrix to the GPU except the first panel. */
        magma_csetmatrix_async( m, n-nb,
                                A(0,nb), 0, lda,
                                dA(0,nb), ldda, queue[0], NULL );

        old_i = 0;
        old_ib = nb;
        for (i = 0; i < k-nb; i += nb) {
            ib = min(k-i, nb);
            if (i>0) {
                /* download i-th panel */
                magma_queue_sync( queue[1] ); 
                magma_cgetmatrix_async( m-i, ib,
                                        dA(i,i), ldda,
                                        A(i,i), 0, lda, queue[0], NULL );

                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_clarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, n-old_i-2*old_ib, old_ib,
                                  dA(old_i, old_i),          ldda, dT, dT_offset,    nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork, dwork_offset, lddwork, queue[1]);

                magma_cgetmatrix_async( i, ib,
                                        dA(0,i), ldda,
                                        A(0,i), 0, lda, queue[1], NULL );
                magma_queue_sync( queue[0] );
            }

            magma_int_t rows = m-i;
            lapackf77_cgeqrf(&rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info);

            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_clarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib, A(i,i), &lda, tau+i, work, &ib);

            cpanel_to_q(MagmaUpper, ib, A(i,i), lda, work+ib*ib);

            /* download the i-th V matrix */
            magma_csetmatrix_async( rows, ib, A(i,i), 0, lda, dA(i,i), ldda, queue[0], NULL );

            /* download the T matrix */
            magma_queue_sync( queue[1] );
            magma_csetmatrix_async( ib, ib, work, 0, ib, dT, dT_offset, nb, queue[0], NULL );
            magma_queue_sync( queue[0] );

            if (i + ib < n) {

                if (i+ib < k-nb) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left (look-ahead) */
                    magma_clarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dT, dT_offset,   nb,
                                      dA(i, i+ib), ldda, dwork, dwork_offset, lddwork, queue[1]);
                    cq_to_panel(MagmaUpper, ib, A(i,i), lda, work+ib*ib);
                }
                else {
                    /* After last panel, update whole trailing matrix. */
                    /* Apply H' to A(i:m,i+ib:n) from the left */
                    magma_clarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib,
                                      dA(i, i   ), ldda, dT, dT_offset,   nb,
                                      dA(i, i+ib), ldda, dwork, dwork_offset, lddwork, queue[1]);
                    cq_to_panel(MagmaUpper, ib, A(i,i), lda, work+ib*ib);
                }

                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }
    
    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib = n-i;
        if (i != 0) {
           magma_cgetmatrix( m, ib, dA(0,i), ldda, A(0,i), 0, lda, queue[1] );
        }
        magma_int_t rows = m-i;
        lapackf77_cgeqrf(&rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info);
    }

    magma_queue_sync(queue[0]);
    magma_queue_sync(queue[1]);
    magma_free( dA );
    
    return *info;
} /* magma_cgeqrf */

