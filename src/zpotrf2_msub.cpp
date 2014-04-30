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

#include "trace.h"

#define Alo(i, j)  (a   +              ((j)+off_j)*lda  + (nb*(((i)/nb)%h)+off_i))
#define Aup(i, j)  (a   +  (nb*(((j)/nb)%h)+off_j)*lda  +               (i+off_i))
#define Alo_off(i, j)   a,             ((j)+off_j)*lda  + (nb*(((i)/nb)%h)+off_i)
#define Aup_off(i, j)   a, (nb*(((j)/nb)%h)+off_j)*lda  +               (i+off_i)

#define dlA(id, i, j)     d_lA[(id)], ((j)*ldda + (i))
#define dlA_offset(i, j)  ((j)*ldda + (i))
#define dlP(id, i, j, k)  d_lP[(id)], ((k)*nb*lddp + (j)*lddp + (i))
#define dlPT(id, i, j, k) d_lP[(id)], ((k)*nb*lddp + (j)*nb   + (i))
#define dlPT_offset(i, j, k) ((k)*nb*lddp + (j)*nb   + (i))
#define dlP_offset(i, j ,k)  ((k)*nb*lddp + (j)*lddp + (i))

extern "C" magma_err_t
magma_zpotrf2_msub(int num_subs, int num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
                   magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                   magmaDoubleComplex_ptr *d_lA, size_t d_lA_offset, magma_int_t ldda, 
                   magmaDoubleComplex_ptr *d_lP, magma_int_t lddp, 
                   magmaDoubleComplex *a, magma_int_t lda, magma_int_t h,
                   magma_int_t *info, magma_queue_t *queues )
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose   
    =======   
    ZPOTRF computes the Cholesky factorization of a complex Hermitian   
    positive definite matrix dA.   

    The factorization has the form   
       dA = U**H * U,  if UPLO = 'U', or   
       dA = L  * L**H,  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   
    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of dA is stored;   
            = 'L':  Lower triangle of dA is stored.   

    N       (input) INTEGER   
            The order of the matrix dA.  N >= 0.   

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)   
            On entry, the Hermitian matrix dA.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of dA contains the upper   
            triangular part of the matrix dA, and the strictly lower   
            triangular part of dA is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of dA contains the lower   
            triangular part of the matrix dA, and the strictly upper   
            triangular part of dA is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization dA = U**H * U or dA = L * L**H.   

    LDDA     (input) INTEGER   
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


    int tot_subs = num_subs*num_gpus;
    magma_int_t     j, jb, nb0, nb2, dd, d, id, j_local, j_local2;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    magmaDoubleComplex_ptr dlpanel;
    size_t dlpanel_offset;
    magma_int_t n_local[MagmaMaxSubs * MagmaMaxGPUs], ldpanel;

    // initialize trace
    trace_init(1, num_gpus, 2, queues);

    *info = 0;
    if ( (uplo != MagmaUpper) && (uplo != MagmaLower) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if ((uplo != MagmaUpper) && tot_subs*ldda < max(1,n)) {
        *info = -4;
    } else if ((uplo == MagmaUpper) && ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    for (d=0; d<tot_subs; d++) {
        /* local-n and local-ld */
        if (uplo == MagmaUpper) {
            n_local[d] = ((n/nb)/tot_subs)*nb;
            if (d < (n/nb)%tot_subs)
               n_local[d] += nb;
            else if (d == (n/nb)%tot_subs)
              n_local[d] += n%nb;
        } else {
            n_local[d] = ((m/nb)/tot_subs)*nb;
            if (d < (m/nb)%tot_subs)
               n_local[d] += nb;
            else if (d == (m/nb)%tot_subs)
              n_local[d] += m%nb;
        }
    }

    /* Use blocked code. */
    if (uplo == MagmaUpper) {
        /* ---------------------------------------------- */
        /* Upper-triangular case                          */
        /* > Compute the Cholesky factorization A = U'*U. */
        /* ---------------------------------------------- */
        for (j=0; j<m; j+=nb) {
            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%tot_subs;
            /* Set the local index where the current panel is */
            j_local = j/(nb*tot_subs);
            jb = min(nb, (m-j));
            if (j > 0) {
                // Wait for the column on CPU
                magma_queue_sync(queues[2*(id%num_gpus)]); 
                /* broadcast off-diagonal column to all gpus */
                d = (j/nb+1)%num_gpus;
                for (dd=0; dd<num_gpus; dd++) {
                    if (d != id%num_gpus) {
                        magma_zsetmatrix_async( j, jb, 
                                                Aup_off(0,j),            lda, 
                                                dlP(d,jb,0,id%num_gpus), lddp, 
                                                queues[2*d], 
                                                trace_gpu_event(d, 0, "set", "set-col") );
                    }
                    d = (d+1)%num_gpus;
                }
                /* Update the current diagonal block */
                trace_gpu_start(id%num_gpus, 1, "herk", "herk");
                magma_zherk(MagmaUpper, MagmaConjTrans, jb, j, 
                            d_neg_one, dlA(id, 0, nb*j_local), ldda,
                            d_one,     dlA(id, j, nb*j_local), ldda,
                            queues[2*(id%num_gpus)+1]);
                magma_queue_sync(queues[2*(id%num_gpus)+1]); // Wait for syrk
            }
            /* Send the diagonal to cpu */
            magma_zgetmatrix_async( jb, jb, 
                                    dlA(id, j, nb*j_local), ldda,
                                    Aup_off(j,j),           lda,
                                    queues[2*(id%num_gpus)], 
                                    trace_gpu_event(id%num_gpus, 0, "get", "get-diag") );
            if (j > 0) {
                /* Compute the local block column of the panel. */
                d = (j/nb+1)%tot_subs;
                for (dd=0; dd<tot_subs; dd++) {
                    j_local2 = j_local+1;
                    if (d > id) j_local2 --;
                    nb0 = nb*j_local2;
                    if (n_local[d] > nb0) {
                        if (d%num_gpus != id%num_gpus) {
                            dlpanel = d_lP[d%num_gpus];
                            dlpanel_offset = dlP_offset(jb, 0, id%num_gpus);
                            ldpanel = lddp;
                            /* Wait for the offdiagonal column */
                            if (dd < num_gpus) magma_queue_sync(queues[2*(d%num_gpus)]);
                        } else {
                            dlpanel = d_lA[id];
                            dlpanel_offset = dlA_offset(0, nb*j_local);
                            ldpanel = ldda;
                        }
                        /* update the panel */
                        trace_gpu_start(d%num_gpus, 1, "gemm", "gemm");
                        magma_zgemm(MagmaConjTrans, MagmaNoTrans, 
                                    jb, n_local[d]-nb0, j, 
                                    c_neg_one, dlpanel, dlpanel_offset, ldpanel,
                                               dlA(d, 0, nb0), ldda, 
                                    c_one,     dlA(d, j, nb0), ldda,
                                    queues[2*(d%num_gpus)+1]);
                    }
                    d = (d+1)%tot_subs;
                }
            }
            /* factor the diagonal */
            magma_queue_sync( queues[2*(id%num_gpus)] ); // wait for the diagonal
            trace_cpu_start(0, "potrf", "potrf");
            lapackf77_zpotrf(MagmaUpperStr, &jb, Aup(j,j), &lda, info);
            trace_cpu_end(0);
            if (*info != 0) {
                *info = *info + j;
                break;
            }

            /* send the diagonal to gpus */
            if ((j+jb) < n) {
                d = (j/nb+1)%num_gpus;
                for (dd=0; dd<num_gpus; dd++) {
                    if (d == id%num_gpus) {
                        dlpanel = d_lA[id];
                        dlpanel_offset = dlA_offset(j, nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = d_lP[d];
                        dlpanel_offset = dlP_offset(0, 0, id%num_gpus);
                        ldpanel = lddp;
                    }
                    magma_zsetmatrix_async( jb, jb, 
                                            Aup_off(j,j),            lda,
                                            dlpanel, dlpanel_offset, ldpanel, 
                                            queues[2*d], 
                                            trace_gpu_event(d, 0, "set", "set-diag"));
                    d = (d+1)%num_gpus;
                }
            } else {
                magma_zsetmatrix_async( jb, jb, 
                                        Aup_off(j,j),           lda, 
                                        dlA(id, j, nb*j_local), ldda,
                                        queues[2*(id%num_gpus)], 
                                        trace_gpu_event(id%num_gpus, 0, "set", "set-diag") );
            }

            /* panel-factorize the off-diagonal */
            if ((j+jb) < n) {
                d = (j/nb+1)%tot_subs;
                for (dd=0; dd<tot_subs; dd++) {
                    /* next column */
                    j_local2 = j_local+1;
                    if (d > id) j_local2--;
                    if (d%num_gpus == id%num_gpus) {
                        dlpanel = d_lA[id];
                        dlpanel_offset = dlA_offset(j, nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = d_lP[d%num_gpus];
                        dlpanel_offset = dlP_offset(0, 0, id%num_gpus);
                        ldpanel = lddp;
                    }
                    nb2 = n_local[d]-nb*j_local2;
                    nb0 = min(nb, nb2);
                    if (dd < num_gpus) magma_queue_sync( queues[2*(d%num_gpus)] ); // wait for the diagonal
                    if (j+jb < m && d == (j/nb+1)%tot_subs) {
                        /* owns the next column, look-ahead the column */
                        trace_gpu_start(d%num_gpus, 1, "trsm", "trsm");
                        magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                     jb, nb0, c_one,
                                     dlpanel, dlpanel_offset, ldpanel,
                                     dlA(d, j, nb*j_local2), ldda, 
                                     queues[2*(d%num_gpus)+1] );
                        /* send the column to cpu */
                        magma_queue_sync(queues[2*(d%num_gpus)+1]);  // wait for lookahead
                        magma_zgetmatrix_async( (j+jb), nb0, 
                                                dlA(d, 0, nb*j_local2), ldda, 
                                                Aup_off(0,j+jb),        lda,
                                                queues[2*(d%num_gpus)], 
                                                trace_gpu_event(d%num_gpus, 0, "get", "get-col") );
                        /* update the remaining blocks */
                        nb2 = nb2 - nb0;
                        trace_gpu_start(d%num_gpus, 1, "trsm", "trsm");
                        magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                     jb, nb2, c_one, 
                                     dlpanel, dlpanel_offset, ldpanel,
                                     dlA(d, j, nb*j_local2+nb0), ldda, 
                                     queues[2*(d%num_gpus)+1] );
                    } else if (nb2 > 0) {
                        /* update the entire trailing matrix */
                        trace_gpu_start(d%num_gpus, 1, "trsm", "trsm");
                        magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, 
                                     jb, nb2, c_one, 
                                     dlpanel, dlpanel_offset, ldpanel,
                                     dlA(d, j, nb*j_local2), ldda,
                                     queues[2*(d%num_gpus)+1] );
                    }
                    d = (d+1)%tot_subs;
                }
            }
        }
    } else { 
        /* -------------------------------------------- */
        /* Lower-triangular case                        */
        /* Compute the Cholesky factorization A = L*L'. */
        /* -------------------------------------------- */
        for (j=0; j<n; j+=nb) {
            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%tot_subs;
            /* Set the local index where the current panel is */
            j_local = j/(nb*tot_subs);
            jb = min(nb, (n-j));
            if (j > 0) {
                if (num_gpus > 1) {
                    // Wait for the row on CPU to broadcast
                    magma_queue_sync(queues[2*(id%num_gpus)]); 
                }
                /* broadcast off-diagonal row to all the GPUs */
                d = (j/nb+1)%num_gpus;
                for (dd=0; dd<num_gpus; dd++) {
                    if (d != id%num_gpus) {
                        /* send it to GPU-d */
                        magma_zsetmatrix_async( jb, j,
                                                Alo_off(j,0),             lda,
                                                dlPT(d,0,jb,id%num_gpus), nb, 
                                                queues[2*d], 
                                                trace_gpu_event(d, 0, "set", "set-row") );
                    }
                    d = (d+1)%num_gpus;
                }
                /* Update the current diagonal block */
                trace_gpu_start(id%num_gpus, 1, "herk", "herk");
                magma_zherk(MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dlA(id, nb*j_local, 0), ldda,
                            d_one,     dlA(id, nb*j_local, j), ldda,
                            queues[2*(id%num_gpus)+1]);
                magma_queue_sync(queues[2*(id%num_gpus)+1]); // wait for syrk
            }
            /* send the diagonal to cpu */
            magma_zgetmatrix_async( jb, jb,
                                    dlA(id, nb*j_local, j), ldda,
                                    Alo_off(j,j),           lda, 
                                    queues[2*(id%num_gpus)], 
                                    trace_gpu_event(id%num_gpus, 0, "get", "get") );
            /* update the offdiagonal blocks */
            if (j > 0) {
                /* compute the block-rows of the panel */
                d = (j/nb+1)%tot_subs;
                for (dd=0; dd<tot_subs; dd++) {
                    j_local2 = j_local+1;
                    if (d > id) j_local2 --;
                    nb0 = nb*j_local2;
                    if (nb0 < n_local[d]) {
                        if (d%num_gpus != id%num_gpus) {
                            dlpanel = d_lP[d%num_gpus];
                            dlpanel_offset = dlPT_offset(0, jb, id%num_gpus);
                            ldpanel = nb;
                            /* Wait for offdiagonal row */
                            if (dd < num_gpus) magma_queue_sync(queues[2*(d%num_gpus)]);
                        } else {
                            dlpanel = d_lA[id];
                            dlpanel_offset = dlA_offset(nb*j_local, 0);
                            ldpanel = ldda;
                        }
                        /* Update the panel */
                        trace_gpu_start(d%num_gpus, 1, "gemm", "gemm");
                        magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                     n_local[d]-nb0, jb, j,
                                     c_neg_one, dlA(d, nb0, 0), ldda,
                                                dlpanel, dlpanel_offset, ldpanel,
                                     c_one,     dlA(d, nb0, j), ldda, 
                                     queues[2*(d%num_gpus)+1]);
                    }
                    d = (d+1)%tot_subs;
                }
            }

            /* factor the diagonal */
            magma_queue_sync( queues[2*(id%num_gpus)] );
            trace_cpu_start(0, "potrf", "potrf");
            lapackf77_zpotrf(MagmaLowerStr, &jb, Alo(j,j), &lda, info);
            trace_cpu_end(0);
            if (*info != 0) {
                printf( " zpotrf returned %d (id=%d,j=%d,j_local=%d,jb=%d)\n",*info,id,j,j_local,jb );
                *info = *info + j;
                break;
            }

            /* send the diagonal to gpus */
            if ((j+jb) < m) {
                d = (j/nb+1)%num_gpus;
                for (dd=0; dd<num_gpus; dd++) {
                    if (d == id%num_gpus) {
                        dlpanel = d_lA[id];
                        dlpanel_offset = dlA_offset(nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = d_lP[d];
                        dlpanel_offset = dlPT_offset(0, 0, id%num_gpus);
                        ldpanel = nb;
                    }
                    magma_zsetmatrix_async( jb, jb,
                                            Alo_off(j,j), lda,
                                            dlpanel,      dlpanel_offset, ldpanel, 
                                            queues[2*d], 
                                            trace_gpu_event(d, 0, "set", "set-diag") );
                    d = (d+1)%num_gpus;
                }
            } else {
                magma_zsetmatrix_async( jb, jb,
                                        Alo_off(j,j),           lda,
                                        dlA(id, nb*j_local, j), ldda, 
                                        queues[2*(id%num_gpus)],
                                        trace_gpu_event(id%num_gpus, 0, "set", "set-diag") );
            }

            /* factorize off-diagonal blocks */
            if ((j+jb) < m) {
                d = (j/nb+1)%tot_subs;
                for (dd=0; dd<tot_subs; dd++) {
                    /* next column */
                    j_local2 = j_local+1;
                    if (d > id) j_local2--;
                    if (d%num_gpus == id%num_gpus) {
                        dlpanel = d_lA[id];
                        dlpanel_offset = dlA_offset(nb*j_local, j);
                        ldpanel = ldda;
                    } else {         
                        dlpanel = d_lP[d%num_gpus];
                        dlpanel_offset = dlPT_offset(0, 0, id%num_gpus);
                        ldpanel = nb;
                    }
                    nb2 = n_local[d] - j_local2*nb;
                    nb0 = min(nb, nb2 );
                    // wait for the diagonal
                    if (dd < num_gpus) magma_queue_sync(queues[2*(d%num_gpus)]);
                    if (j+jb < n && d == (j/nb+1)%tot_subs) {
                        /* owns the next column, look-ahead the column */
                        trace_gpu_start(d%num_gpus, 1, "trsm", "trsm");
                        magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                     nb0, jb, c_one,
                                     dlpanel,  dlpanel_offset, ldpanel, 
                                     dlA(d, nb*j_local2, j), ldda,
                                     queues[2*(d%num_gpus)+1]);
                        /* send the column to cpu */
                        magma_queue_sync( queues[2*(d%num_gpus)+1] ); // wait for lookahead
                        magma_zgetmatrix_async( nb0, j+jb,
                                                dlA(d, nb*j_local2, 0), ldda,
                                                Alo_off(j+jb,0),        lda, 
                                                queues[2*(d%num_gpus)], 
                                                trace_gpu_event(d%num_gpus, 0, "get", "get") );
                        /* update the remaining blocks */
                        nb2 = nb2 - nb0;
                        trace_gpu_start(d%num_gpus, 1, "trsm", "trsm");
                        magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                     nb2, jb, c_one,
                                     dlpanel, dlpanel_offset, ldpanel, 
                                     dlA(d, nb*j_local2+nb0, j), ldda, 
                                     queues[2*(d%num_gpus)+1]);
                    } else if (nb2 > 0) {
                        /* update the entire trailing matrix */
                        trace_gpu_start(d%num_gpus, 1, "trsm", "trsm");
                        magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                     nb2, jb, c_one,
                                     dlpanel, dlpanel_offset, ldpanel, 
                                     dlA(d, nb*j_local2, j), ldda, 
                                     queues[2*(d%num_gpus)+1]);
                    }
                    d = (d+1)%tot_subs;
                }
            }
        }
    } /* end of else not upper */

    /* clean up */
    for( d=0; d<num_gpus; d++ ) {
        magma_queue_sync( queues[2*d] );
        magma_queue_sync( queues[2*d+1] );
    }

    trace_finalize("zpotrf_msub.svg", "trace.css");
    return *info;
} /* magma_zpotrf2_msub */
