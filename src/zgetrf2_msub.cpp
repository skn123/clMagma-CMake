/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @precisions normal z -> s d c

*/
#include <math.h>
#include "common_magma.h"
magma_err_t
magma_ztranspose_host( int flag, int d,
    magmaDoubleComplex_ptr odata, int offseto, int ldo,
    magmaDoubleComplex_ptr idata, int offseti, int ldi,
    int m, int n,
    magma_queue_t queue )
{
    int i, j;
    magmaDoubleComplex *work1, *work2;
    if (MAGMA_SUCCESS != magma_zmalloc_cpu( &work1, m*n )) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    if (MAGMA_SUCCESS != magma_zmalloc_cpu( &work2, m*n )) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    /* download to CPU */
    magma_zgetmatrix_async( m, n,
                            idata, offseti, ldi,
                            work1, 0, m, queue, NULL );
    magma_queue_sync( queue );

    /* transpose it on CPU */
    for( i=0; i<m; i++ ) {
       for( j=0; j<n; j++ ) work2[j+i*n] = work1[i+j*m];
    }

    /* upload to GPU */
    magma_zsetmatrix_async( n, m,
                            work2, 0, n,
                            odata, offseto, ldo, 
                            queue, NULL );

    magma_queue_sync( queue );
    magma_free_cpu( work2 );
    magma_free_cpu( work1 );
    return 0;
}

extern "C" magma_err_t
magma_zgetrf2_msub(magma_int_t num_subs, magma_int_t num_gpus, 
         magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
         magmaDoubleComplex_ptr *d_lAT, size_t dlAT_offset, magma_int_t lddat, 
         magma_int_t *ipiv,
         magmaDoubleComplex_ptr *d_panel, 
         magmaDoubleComplex_ptr *d_lAP, size_t dlAP_offset, 
         magmaDoubleComplex *w, magma_int_t ldw,
         magma_int_t *info, magma_queue_t *queues)
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose
    =======

    ZGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    Use two buffer to send panels..

    Arguments
    =========

    NUM_GPUS 
            (input) INTEGER
            The number of GPUS to be used for the factorization.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDDA     (input) INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  if INFO = -7, internal GPU memory allocation failed.
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.
    =====================================================================    */

#define inAT(id,i,j)  d_lAT[(id)], (((offset)+(i)*nb)*lddat + (j)*nb)
#define inAT_offset(i, j) (((offset)+(i)*nb)*lddat + (j)*nb)
#define W(j)     (w +((j)%(1+num_gpus))*nb*ldw)
#define W_off(j)  w, ((j)%(1+num_gpus))*nb*ldw

    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    magma_int_t tot_subs = num_subs * num_gpus;
    magma_int_t block_size = 32;
    magma_int_t iinfo, maxm, mindim;
    magma_int_t i, d, dd, rows, cols, s;
    magma_int_t id, i_local, i_local2, nb0, nb1;

    /* local submatrix info */
    magma_int_t ldpan[MagmaMaxSubs * MagmaMaxGPUs],
                n_local[MagmaMaxSubs * MagmaMaxGPUs]; 
    size_t panel_local_offset[MagmaMaxSubs * MagmaMaxGPUs];
    magmaDoubleComplex_ptr panel_local[MagmaMaxSubs * MagmaMaxGPUs];

    /* Check arguments */
    *info = 0;
    if (m < 0)
    *info = -2;
    else if (n < 0)
    *info = -3;
    else if (tot_subs*lddat < max(1,n))
    *info = -5;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    mindim = min(m, n);
    if (tot_subs > ceil((double)n/nb)) {
      *info = -1;
      return *info;
    }
    
    else{
      /* Use hybrid blocked code. */
      maxm  = ((m + block_size-1)/block_size)*block_size;

      /* some initializations */
      for (i=0; i<tot_subs; i++) {
        n_local[i] = ((n/nb)/tot_subs)*nb;
        if (i < (n/nb)%tot_subs)
           n_local[i] += nb;
        else if (i == (n/nb)%tot_subs)
           n_local[i] += n%nb;
      }

      /* start sending the first panel to cpu */
      nb0 = min(mindim, nb);
      if (nb0 == nb) {
        magma_ztranspose(  d_lAP[0], dlAP_offset, maxm, inAT(0,0,0), lddat, nb0, maxm, queues[2*0+1] );
      } else {
        magma_ztranspose2( d_lAP[0], dlAP_offset, maxm, inAT(0,0,0), lddat, nb0, maxm, queues[2*0+1] );
      }
      magma_zgetmatrix_async( m, nb0,
                              d_lAP[0], dlAP_offset, maxm,
                              W_off(0), ldw, queues[2*0+1], NULL );
      clFlush(queues[2*0+1]);
      /* ------------------------------------------------------------------------------------- */

      s = mindim / nb;
      for (i=0; i<s; i++) {
          /* Set the submatrix ID that holds the current panel */
          id = i%tot_subs;

          /* Set the local index where the current panel is */
          i_local = i/tot_subs;
          // cols for gpu panel
          cols  = maxm - i*nb;
          // rows for cpu panel
          rows  = m - i*nb;

          /* synchrnoize i-th panel from id-th gpu into work */
          magma_queue_sync( queues[2*(id%num_gpus)+1] );

          /* i-th panel factorization */
          lapackf77_zgetrf( &rows, &nb, W(i), &ldw, ipiv+i*nb, &iinfo);
          if ((*info == 0) && (iinfo > 0)) {
              *info = iinfo + i*nb;
              //break;
          }

          /* start sending the panel to all the gpus */
          d = (i+1)%num_gpus;
          for (dd=0; dd<num_gpus; dd++) {
              magma_zsetmatrix_async( rows, nb,
                                      W_off(i), ldw,
                                      d_lAP[d], dlAP_offset+(i%(2+num_gpus))*nb*maxm, maxm, 
                                      queues[2*d+1], NULL );
              d = (d+1)%num_gpus;
          }
          /* apply the pivoting */
          d = (i+1)%tot_subs;
          for (dd=0; dd<tot_subs; dd++) {
              if (dd == 0) {
                // row offset will be added to ipiv in long2  
                magma_zpermute_long2( lddat, inAT(d,0,0), lddat, ipiv, nb, i*nb, queues[2*(d%num_gpus)] );
              } else {
                // ipiv is already added by row offset, calling long3   
                magma_zpermute_long3( lddat, inAT(d,0,0), lddat, ipiv, nb, i*nb, queues[2*(d%num_gpus)] );
              }
              d = (d+1)%tot_subs;
          }

          /* update the trailing-matrix/look-ahead */
          d = (i+1)%tot_subs;
          for (dd=0; dd<tot_subs; dd++) {
              /* storage for panel */
              if (d%num_gpus == id%num_gpus) {
                  /* the panel belond to this gpu */
                  panel_local[d] = d_lAT[id];
                  panel_local_offset[d] = inAT_offset(i, i_local);
                  ldpan[d] = lddat;
                  /* next column */
                  i_local2 = i_local;
                  if( d <= id ) i_local2 ++;
              } else {
                  /* the panel belong to another gpu */
                  panel_local[d] = d_panel[d%num_gpus];  
                  panel_local_offset[d] = (i%(2+num_gpus))*nb*maxm;
                  ldpan[d] = nb;
                  /* next column */
                  i_local2 = i_local;
                  if( d < id ) i_local2 ++;
              }
              /* the size of the next column */
              if (s > (i+1)) {
                  nb0 = nb;
              } else {
                  nb0 = n_local[d]-nb*(s/tot_subs);
                  if(d < s%tot_subs) nb0 -= nb;
              }
              if (d == (i+1)%tot_subs) {
                  /* owns the next column, look-ahead the column */
                  nb1 = nb0;
              } else {
                  /* update the entire trailing matrix */
                  nb1 = n_local[d] - i_local2*nb;
              }
              
              /* gpu updating the trailing matrix */
              if (d == (i+1)%tot_subs) { /* look-ahead, this is executed first (i.e., dd=0)  */
                  magma_queue_sync(queues[2*(d%num_gpus)]);   /* pivoting done? (overwrite with panel) */
                  magma_ztranspose(panel_local[d], panel_local_offset[d], ldpan[d], 
                                   d_lAP[d%num_gpus], dlAP_offset+(i%(2+num_gpus))*nb*maxm, maxm, cols, nb, queues[2*(d%num_gpus)+1]);
                  magma_queue_sync(queues[2*(d%num_gpus)+1]); /* panel arrived and transposed for remaining update ? */

                  magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                               nb1, nb, c_one,
                               panel_local[d], panel_local_offset[d], ldpan[d],
                               inAT(d, i, i_local2), lddat, queues[2*(d%num_gpus)+1]);

                  magma_zgemm( MagmaNoTrans, MagmaNoTrans, 
                               nb1, m-(i+1)*nb, nb, 
                               c_neg_one, inAT(d, i,   i_local2),         lddat,
                                          panel_local[d], panel_local_offset[d]+nb*ldpan[d], ldpan[d], 
                               c_one,     inAT(d, i+1, i_local2),         lddat,
                               queues[2*(d%num_gpus)+1]);
              } else { /* no look-ahead */
                  if (dd < num_gpus) {
                      /* synch and transpose only the first time */
                      magma_queue_sync(queues[2*(d%num_gpus)+1]); /* panel arrived? */
                      magma_ztranspose(panel_local[d], panel_local_offset[d], ldpan[d], 
                                       d_lAP[d%num_gpus], dlAP_offset+(i%(2+num_gpus))*nb*maxm, maxm, cols, nb, queues[2*(d%num_gpus)]);
                  }

                  magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                               nb1, nb, c_one,
                               panel_local[d], panel_local_offset[d], ldpan[d],
                               inAT(d, i, i_local2), lddat, queues[2*(d%num_gpus)]);
              
                  magma_zgemm( MagmaNoTrans, MagmaNoTrans, 
                               nb1, m-(i+1)*nb, nb, 
                               c_neg_one, inAT(d, i,   i_local2),         lddat,
                                          panel_local[d], panel_local_offset[d]+nb*ldpan[d], ldpan[d], 
                               c_one,     inAT(d, i+1, i_local2),         lddat,
                               queues[2*(d%num_gpus)]);    
              }
              if (d == (i+1)%tot_subs) {
                  /* Set the local index where the current panel is */
                  int loff    = i+1;
                  int i_local = (i+1)/tot_subs;
                  int ldda    = maxm - (i+1)*nb;
                  int cols    = m - (i+1)*nb;
                  nb0 = min(nb, mindim - (i+1)*nb); /* size of the diagonal block */
                  
                  if (nb0 > 0) {
                      /* transpose the panel for sending it to cpu */
                      if (i+1 < s) {
                          magma_ztranspose(  d_lAP[d%num_gpus], dlAP_offset + ((i+1)%(2+num_gpus))*nb*maxm, ldda, 
                                             inAT(d,loff,i_local), lddat, nb0, ldda, queues[2*(d%num_gpus)+1] );
                      } else {
                          magma_ztranspose2( d_lAP[d%num_gpus], dlAP_offset + ((i+1)%(2+num_gpus))*nb*maxm, ldda, 
                                             inAT(d,loff,i_local), lddat, nb0, ldda, queues[2*(d%num_gpus)+1] );
                      }
                
                      /* send the panel to cpu */
                      magma_zgetmatrix_async( cols, nb0, 
                                              d_lAP[d%num_gpus], dlAP_offset + ((i+1)%(2+num_gpus))*nb*maxm, ldda, 
                                              W_off(i+1), ldw, queues[2*(d%num_gpus)+1], NULL );
                  }
              } else {
                  //trace_gpu_end( d, 0 );
              }
              d = (d+1)%tot_subs;
          }

          /* update the remaining matrix by gpu owning the next panel */
          if ((i+1) < s) {
              d = (i+1)%tot_subs;
              int i_local = (i+1)/tot_subs;
              int rows  = m - (i+1)*nb;
              
              magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                           n_local[d] - (i_local+1)*nb, nb, 
                           c_one, panel_local[d], panel_local_offset[d], ldpan[d], 
                                  inAT(d,i,i_local+1), lddat, queues[2*(d%num_gpus)] );
                  
              magma_zgemm( MagmaNoTrans, MagmaNoTrans, 
                           n_local[d]-(i_local+1)*nb, rows, nb, 
                           c_neg_one, inAT(d,i,i_local+1), lddat, 
                                      panel_local[d], panel_local_offset[d]+nb*ldpan[d], ldpan[d], 
                           c_one,     inAT(d,i+1,  i_local+1), lddat, queues[2*(d%num_gpus)] );
          }
      } /* end of for i=1..s */
      /* ------------------------------------------------------------------------------ */

      /* Set the GPU number that holds the last panel */
      id = s%tot_subs;

      /* Set the local index where the last panel is */
      i_local = s/tot_subs;

      /* size of the last diagonal-block */
      nb0 = min(m - s*nb, n - s*nb);
      rows = m    - s*nb;
      cols = maxm - s*nb;

      if (nb0 > 0) {

          /* wait for the last panel on cpu */
          magma_queue_sync( queues[2*(id%num_gpus)+1] );
          
          /* factor on cpu */
          lapackf77_zgetrf( &rows, &nb0, W(s), &ldw, ipiv+s*nb, &iinfo );
          if ( (*info == 0) && (iinfo > 0) )
              *info = iinfo + s*nb;

          /* send the factor to gpus */
          for (d=0; d<num_gpus; d++) {
              magma_zsetmatrix_async( rows, nb0, W_off(s), ldw,
                                      d_lAP[d], dlAP_offset+(s%(2+num_gpus))*nb*maxm, cols, 
                                      queues[2*d+1], NULL );
          }

          for (d=0; d<tot_subs; d++) {
              if (d == 0) {
                  magma_zpermute_long2( lddat, inAT(d,0,0), lddat, ipiv, nb0, s*nb, queues[2*(d%num_gpus)] );
              } else {
                  magma_zpermute_long3( lddat, inAT(d,0,0), lddat, ipiv, nb0, s*nb, queues[2*(d%num_gpus)] );
              }
          }

          d = id;
          for (dd=0; dd<tot_subs; dd++) {
              /* wait for the pivoting to be done */
              if (dd < num_gpus) {
                  /* synch only the first time */
                  magma_queue_sync( queues[2*(d%num_gpus)] );
              }

              i_local2 = i_local;
              if (d%num_gpus == id%num_gpus) {
                  /* the panel belond to this gpu */
                  panel_local[d] = d_lAT[id];
                  panel_local_offset[d] = inAT_offset(s, i_local);
                  if (dd < num_gpus) {
                      magma_ztranspose2( panel_local[d], panel_local_offset[d], lddat, 
                                         d_lAP[d%num_gpus], dlAP_offset+(s%(2+num_gpus))*nb*maxm, cols, 
                                         rows, nb0, queues[2*(d%num_gpus)+1]);
                  }
                  /* size of the "extra" block */
                  if (d == id) { /* the last diagonal block belongs to this submatrix */
                     nb1 = nb0;
                  } else if (d < id) {
                     nb1 = nb;
                  } else {
                     nb1 = 0;
                  }
                  if (n_local[d] > i_local*nb+nb1) {
                      magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                                   n_local[d] - (i_local*nb+nb1), nb0, c_one,
                                   panel_local[d], panel_local_offset[d], lddat, 
                                   inAT(d, s, i_local)+nb1, lddat, queues[2*(d%num_gpus)+1]);
                  }
              } else if (n_local[d] > i_local2*nb) {
                  /* the panel belong to another gpu */
                  panel_local[d] = d_panel[d%num_gpus];
                  panel_local_offset[d] = (s%(2+num_gpus))*nb*maxm;

                  /* next column */
                  if (d < num_gpus) {
                      /* transpose only the first time */
                      magma_ztranspose2( panel_local[d], panel_local_offset[d], nb, 
                                         d_lAP[d%num_gpus], dlAP_offset+(s%(2+num_gpus))*nb*maxm, cols, 
                                         rows, nb0, queues[2*(d%num_gpus)+1]);
                  }
                  if (d < id) i_local2++;
                  nb1 = n_local[d] - i_local2*nb;
                  magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                               nb1, nb0, c_one,
                               panel_local[d], panel_local_offset[d], nb, 
                               inAT(d,s,i_local2), lddat, queues[2*(d%num_gpus)+1]);
              }
              d = (d+1)%tot_subs;
          }
      } /* if( nb0 > 0 ) */

      /* clean up */
      for (d=0; d<num_gpus; d++) {
          magma_queue_sync( queues[2*d] );
          magma_queue_sync( queues[2*d+1] );
      } 
    }
    return *info;
    /* End of MAGMA_ZGETRF2_MSUB */
}

#undef inAT
