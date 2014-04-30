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

//#define dlA(id, i, j)  (d_lA[id] + (j)*ldda + (i))
//#define dlAT(id, i, j)  (d_lA[id] + (j)*lddat + (i))

//#define dlP(id, i, j)  (d_lP[id] + (j)*lddp + (i))
//#define dlPT(id, i, j)  (d_lP[id] + (j)*nb    + (i))

#define Alo(i, j)  (a   +            ((j)+off_j)*lda  + (nb*(((i)/nb)%h)+off_i))
#define Aup(i, j)  (a   +(nb*(((j)/nb)%h)+off_j)*lda  +               (i+off_i))

#define dlA(id, i, j)     d_lA[(id)], ((j)*ldda + (i))
#define dlA_offset(i, j)  ((j)*ldda + (i))
#define dlP(id, i, j, k)  d_lP[(id)], ((k)*nb*lddp + (j)*lddp + (i))
#define dlPT(id, i, j, k) d_lP[(id)], ((k)*nb*lddp + (j)*nb   + (i))
#define dlPT_offset(i, j, k) ((k)*nb*lddp + (j)*nb   + (i))
//#define dlPT[id] d_lP[id]

extern "C" magma_err_t
magma_zpotrf2_mgpu(int num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
                   magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                   magmaDoubleComplex_ptr *d_lA, size_t d_lA_offset, magma_int_t ldda, 
                   magmaDoubleComplex_ptr *d_lP,  magma_int_t lddp, 
                   magmaDoubleComplex *a,      magma_int_t lda,   magma_int_t h,
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


    magma_int_t     j, jb, nb0, nb2, dd, d, id, j_local, j_local2, buf;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    magmaDoubleComplex_ptr dlpanel;
    size_t dlpanel_offset;
    magma_int_t n_local[MagmaMaxGPUs], ldpanel;
    magma_event_t* events[MagmaMaxGPUs][2];

    *info = 0;
    if ( (uplo != MagmaUpper) && (uplo != MagmaLower) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if ((uplo != MagmaUpper) && num_gpus*ldda < max(1,n)) {
        *info = -4;
    } else if ((uplo == MagmaUpper) && ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    {

      for( d=0; d<num_gpus; d++ ) {
        /* local-n and local-ld */
        if (uplo == MagmaUpper) {
            n_local[d] = ((n/nb)/num_gpus)*nb;
            if (d < (n/nb)%num_gpus)
               n_local[d] += nb;
            else if (d == (n/nb)%num_gpus)
              n_local[d] += n%nb;
        } else {
            n_local[d] = ((m/nb)/num_gpus)*nb;
            if (d < (m/nb)%num_gpus)
               n_local[d] += nb;
            else if (d == (m/nb)%num_gpus)
              n_local[d] += m%nb;
        }
      }

      /* Use blocked code. */
      if (uplo == MagmaUpper) 
        {
          /* ---------------------------------------------- */
          /* Upper-triangular case                          */
          /* > Compute the Cholesky factorization A = U'*U. */
          /* ---------------------------------------------- */
            printf("support Lower case only!\n");
        } else { 
            trace_init(1, num_gpus, 2, queues);
            /* -------------------------------------------- */
            /* Lower-triangular case                        */
            /* Compute the Cholesky factorization A = L*L'. */
            /* -------------------------------------------- */
            // try one queue per device firstly!
            for (j=0; j<n; j+=nb) {

              /* Set the GPU number that holds the current panel */
              id  = (j/nb)%num_gpus;
              buf = (j/nb)%num_gpus;

              /* Set the local index where the current panel is */
              j_local = j/(nb*num_gpus);
              jb = min(nb, (n-j));

              if( j > 0 ) {
                  /* needed on pluto... */
                  //magma_setdevice(id);
                  magma_queue_sync( queues[id*2] ); // wait for the column on CPU

                  /* broadcast offdiagonal row to all gpus */
                  d = (j/nb+1)%num_gpus;
                  for( dd=0; dd<num_gpus; dd++ ) {
                      if( d != id ) {
                          /* wait for it on CPU */
                          //magma_queue_sync( queues[d*2] );

                          /* send it to GPU */
                          trace_gpu_start(d, 0, "comm_set", "set");
                          events[d][0] = trace_gpu_end(d, 0);
                          magma_zsetmatrix_async( jb, j,
                                                  Alo(j,0), 0,      lda,
                                                  dlPT(d,0,jb,buf), nb, 
                                                  queues[d*2], events[d][0] );
                         clFlush(queues[d*2]);
                      }
                      d = (d+1)%num_gpus;
                  }
              }

              /* Update the current diagonal block */
              if( j > 0 ) {
                  trace_gpu_start(id, 1, "herk", "herk");
                  events[id][1] = trace_gpu_end(id, 1);
                  magma_zherk_trace(MagmaLower, MagmaNoTrans, jb, j,
                              d_neg_one, dlA(id, nb*j_local, 0), ldda,
                              d_one,     dlA(id, nb*j_local, j), ldda,
                              queues[id*2+1], events[id][1]);
                 // magma_queue_sync( queues[id*2+1] ); // wait for syrk
                  clWaitForEvents(1, events[id][1]);
              }

              /* send the diagonal to cpu */
              trace_gpu_start(id, 0, "comm_get", "get");
              events[id][0] = trace_gpu_end(id, 0);
              magma_zgetmatrix_async( jb, jb,
                                      dlA(id, nb*j_local, j), ldda,
                                      Alo(j,j), 0,            lda, 
                                      queues[id*2], events[id][0] );
              clFlush(queues[id*2]);


              /* update the offdiagonal blocks */
              if ( j > 0 ) {
                  /* compute the block-rows of the panel */
                  d = (j/nb+1)%num_gpus;
                  for( dd=0; dd<num_gpus; dd++ ) {
                      j_local2 = j_local+1;
                      if( d > id ) j_local2 --;
                      nb0 = nb*j_local2;

                      if( nb0 < n_local[d] ) {
                          if( d != id ) {
                              //dlpanel = dlPT(d);
                              dlpanel = d_lP[d];
                              dlpanel_offset = dlPT_offset(0, jb, buf);
                              ldpanel = nb;

                              /* wait for offdiagonal row */
                              magma_queue_sync(queues[d*2]);
                          } else {
                              dlpanel = d_lA[d];
                              dlpanel_offset = dlA_offset(nb*j_local, 0);
                              ldpanel = ldda;
                          }
                        
                          trace_gpu_start(d, 1, "gemm", "gemm");
                          events[d][1] = trace_gpu_end(d, 1);
                          magma_zgemm_trace( MagmaNoTrans, MagmaConjTrans,
                                       n_local[d]-nb0, jb, j,
                                       c_neg_one, dlA(d, nb0, 0), ldda,
                                                  dlpanel, dlpanel_offset, ldpanel,
                                       c_one,     dlA(d, nb0, j), ldda, 
                                       queues[d*2+1], events[d][1]);
                      }
                      d = (d+1)%num_gpus;
                  }
              }

              /* send the diagonal to cpu */
              /*
              trace_gpu_start(id, 0, "comm_get", "get");
              events[id][0] = trace_gpu_end(id, 0);
              magma_zgetmatrix_async( jb, jb,
                                      dlA(id, nb*j_local, j), ldda,
                                      Alo(j,j), 0,            lda, 
                                      queues[id*2], events[id][0] );
              clFlush(queues[id*2]);
              */
              /* factor the diagonal */
              magma_queue_sync( queues[id*2] );
              trace_cpu_start(0, "potrf", "potrf");
              lapackf77_zpotrf(MagmaLowerStr, &jb, Alo(j,j), &lda, info);
              trace_cpu_end(0);
              if (*info != 0) {
                  printf("row number: %d\n", j);
                  *info = *info + j;
                  break;
              }

              /* send the diagonal to gpus */
              if ( (j+jb) < m ) {
                  d = (j/nb+1)%num_gpus;
                  for( dd=0; dd<num_gpus; dd++ ) {
                      if( d == id ) {
                          dlpanel = d_lA[d];
                          dlpanel_offset = dlA_offset(nb*j_local, j);
                          ldpanel = ldda;
                      } else {
                          //dlpanel = dlPT(d);
                          dlpanel = d_lP[d];
                          dlpanel_offset = dlPT_offset(0, 0, buf);
                          ldpanel = nb;
                      }
                      trace_gpu_start(d, 0, "comm_set", "set");
                      events[d][0] = trace_gpu_end(d, 0);
                      magma_zsetmatrix_async( jb, jb,
                                              Alo(j,j), 0, lda,
                                              dlpanel,  dlpanel_offset, ldpanel, 
                                              queues[d*2], events[d][0] );
                      clFlush(queues[d*2]);
                      d = (d+1)%num_gpus;
                  }
              } else {
                  trace_gpu_start(id, 0, "comm_set", "set");
                  events[id][0] = trace_gpu_end(id, 0);
                  magma_zsetmatrix_async( jb, jb,
                                          Alo(j,j),       0,      lda,
                                          dlA(id, nb*j_local, j), ldda, 
                                          queues[id*2], events[id][0] );
                  clFlush(queues[id*2]);
              }

              /* factorize off-diagonal blocks */
              if ( (j+jb) < m ) {
                  d = (j/nb+1)%num_gpus;
                  for( dd=0; dd<num_gpus; dd++ ) {
                      /* next column */
                      j_local2 = j_local+1;
                      if( d > id ) j_local2--;
                      if( d == id ) {
                          dlpanel = d_lA[d];
                          dlpanel_offset = dlA_offset(nb*j_local, j);
                          ldpanel = ldda;
                      } else {         
                          //dlpanel = dlPT(d);
                          dlpanel = d_lP[d];
                          dlpanel_offset = dlPT_offset(0, 0, buf);
                          ldpanel = nb;
                      }
                      nb2 = n_local[d] - j_local2*nb;
                      nb0 = min(nb, nb2 );
                        
                      magma_queue_sync(queues[d*2]);
                      // wait for the diagonal
                      if( j+jb < n && d == (j/nb+1)%num_gpus ) {
                          /* owns the next column, look-ahead the column */
                          trace_gpu_start(d, 1, "trsm", "trsm");
                          events[d][1] = trace_gpu_end(d, 1);
                          magma_ztrsm_trace( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                       nb0, jb, c_one,
                                       dlpanel,  dlpanel_offset, ldpanel, 
                                       dlA(d, nb*j_local2, j), ldda,
                                       queues[d*2+1], events[d][1]);
                          /* send the column to cpu */
                          if( j+jb < n ) {
                              magma_queue_sync( queues[d*2+1] ); // wait for lookahead
                              trace_gpu_start(d, 0, "comm_get", "get");
                              events[d][0] = trace_gpu_end(d, 0);
                              magma_zgetmatrix_async( nb0, j+jb,
                                                      dlA(d, nb*j_local2, 0), ldda,
                                                      Alo(j+jb,0), 0,           lda, 
                                                      queues[d*2], events[d][0]);
                              clFlush(queues[d*2]);
                          }
                          /* update the remaining blocks */
                          nb2 = nb2 - nb0;
                          trace_gpu_start(d, 1, "trsm", "trsm");
                          events[d][1] = trace_gpu_end(d, 1);
                          magma_ztrsm_trace( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                       nb2, jb, c_one,
                                       dlpanel, dlpanel_offset, ldpanel, 
                                       dlA(d, nb*j_local2+nb0, j), ldda, 
                                       queues[d*2+1], events[d][1]);
                      } else if( nb2 > 0 ) {
                          /* update the entire trailing matrix */
                          trace_gpu_start(d, 1, "trsm", "trsm");
                          events[d][1] = trace_gpu_end(d, 1);
                          magma_ztrsm_trace( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                       nb2, jb, c_one,
                                       dlpanel, dlpanel_offset, ldpanel, 
                                       dlA(d, nb*j_local2, j), ldda, 
                                       queues[d*2+1], events[d][1]);
                      }
                      d = (d+1)%num_gpus;
                  }
              }
            }
          } /* end of else not upper */

          /* clean up */
          for( d=0; d<num_gpus; d++ ) {
              magma_queue_sync( queues[d*2] );
              magma_queue_sync( queues[d*2+1] );
          }
          trace_finalize("dpotrf2_mgpu.svg", "trace.css");

    } /* end of not lapack */

    return *info;
} /* magma_zpotrf_mgpu */
