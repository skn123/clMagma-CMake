/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @author Stan Tomov
       @generated from zgetrf.cpp normal z -> c, Fri Jan 10 15:51:18 2014
*/
#include "common_magma.h"



extern "C" magma_err_t
magma_cgetrf(magma_int_t m, magma_int_t n, magmaFloatComplex *a, magma_int_t lda,
             magma_int_t *ipiv, magma_int_t *info,
             magma_queue_t* queue )
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose
    =======
    CGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.  This version does not
    require work space on the GPU passed as input. GPU memory is allocated
    in the routine.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    If the current stream is NULL, this version replaces it with user defined
    stream to overlap computation with communication. 

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    =====================================================================    */

#define dAT(i,j) dAT, dAT_offset + ((i)*nb*lddat + (j)*nb)

    magmaFloatComplex *work;
    magmaFloatComplex_ptr dAT, dA, dwork, dAP;
    size_t dA_offset, dAT_offset;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t     iinfo, nb;

    *info = 0;

    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    nb = magma_get_cgetrf_nb(m);

    if ( (nb <= 1) || (nb >= min(m,n)) ) {
        /* Use CPU code. */
        lapackf77_cgetrf(&m, &n, a, &lda, ipiv, info);
    } else {
        /* Use hybrid blocked code. */
        magma_int_t maxm, maxn, ldda, maxdim, lddat;
        magma_int_t i, rows, cols, s = min(m, n)/nb;
        
        maxm = ((m + 31)/32)*32;
        maxn = ((n + 31)/32)*32;

        lddat   = maxn;
        ldda    = maxm;

        maxdim = max(maxm, maxn);

        /* set number of GPUs */
        magma_int_t num_gpus = magma_num_gpus();
        if ( num_gpus > 1 ) {
            /* call multi-GPU non-GPU-resident interface  */
            printf("multiple-GPU verison not implemented\n");
            return MAGMA_ERR_NOT_IMPLEMENTED;
            // magma_cgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
            // return *info;
        }

        /* explicitly checking the memory requirement */
        magma_int_t totalMem = magma_queue_meminfo( queue[0] );
        totalMem /= sizeof(magmaFloatComplex);

        int h = 1+(2+num_gpus), num_gpus2 = num_gpus;
        int NB = (magma_int_t)(0.8*totalMem/maxm-h*nb);
        char * ngr_nb_char = getenv("MAGMA_NGR_NB");
        if( ngr_nb_char != NULL ) NB = max( nb, min( NB, atoi(ngr_nb_char) ) );

        if( num_gpus > ceil((float)NB/nb) ) {
            num_gpus2 = (int)ceil((float)NB/nb);
            h = 1+(2+num_gpus2);
            NB = (magma_int_t)(0.8*totalMem/maxm-h*nb);
        } 
        if( num_gpus2*NB < n ) {
            /* require too much memory, so call non-GPU-resident version */
            printf("non-GPU-resident version not implemented\n");
            return MAGMA_ERR_NOT_IMPLEMENTED; 
            //magma_cgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
            //return *info;
        }

        work = a;
        if (maxdim*maxdim < 2*maxm*maxn) {
            // if close to square, allocate square matrix and transpose in-place
            if (MAGMA_SUCCESS != 
                magma_cmalloc( &dwork, (nb*maxm + maxdim*maxdim) ) ) {
                /* alloc failed so call non-GPU-resident version */
                printf("non-GPU-resident version not implemented\n");
                return MAGMA_ERR_NOT_IMPLEMENTED;
                //magma_cgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
                //return *info;
            }
            dAP = dwork;

            dA = dwork;
            dA_offset = nb*maxm;            

            ldda = lddat = maxdim;
            magma_csetmatrix( m, n, a, 0, lda, dA, dA_offset, ldda, queue[0] );
            
            dAT = dA;
            dAT_offset = dA_offset;
            magma_ctranspose_inplace( dAT, dAT_offset, ldda, lddat, queue[0] );
        }
        else {
            // if very rectangular, allocate dA and dAT and transpose out-of-place
            if (MAGMA_SUCCESS != 
                magma_cmalloc( &dwork, (nb + maxn)*maxm )) {
                /* alloc failed so call non-GPU-resident version */
                printf("non-GPU-resident version not implemented\n");
                return MAGMA_ERR_NOT_IMPLEMENTED;
                //magma_cgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
                //return *info;
            }
            dAP = dwork;

            dA = dwork;
            dA_offset = nb*maxm;
            
            magma_csetmatrix( m, n, a, 0, lda, dA, dA_offset, ldda, queue[0] );
            
            if (MAGMA_SUCCESS != magma_cmalloc( &dAT, maxm*maxn )) {
                /* alloc failed so call non-GPU-resident version */
                magma_free( dwork );
                printf("non-GPU-resident version not implemented\n");
                return MAGMA_ERR_NOT_IMPLEMENTED;
                //magma_cgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
                //return *info;
            }
            dAT_offset = 0;   
            magma_ctranspose2( dAT, dAT_offset, lddat, dA, dA_offset, ldda, m, n, queue[0] );
        }
        
        lapackf77_cgetrf( &m, &nb, work, &lda, ipiv, &iinfo);

        for( i = 0; i < s; i++ )
        {
            // download i-th panel
            cols = maxm - i*nb;
            
            if (i>0){
                // download i-th panel 
                magma_ctranspose( dAP, 0, cols, dAT(i,i), lddat, nb, cols, queue[0] );

                magma_queue_sync(queue[0]);
                magma_cgetmatrix_async( m-i*nb, nb, dAP, 0, cols, work, 0, lda, 
                                        queue[1], NULL);
                
                magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n - (i+1)*nb, nb,
                             c_one, dAT(i-1,i-1), lddat,
                                    dAT(i-1,i+1), lddat, queue[0] );
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(i+1)*nb, m-i*nb, nb,
                             c_neg_one, dAT(i-1,i+1), lddat,
                                        dAT(i,  i-1), lddat,
                             c_one,     dAT(i,  i+1), lddat, queue[0] );

                // do the cpu part
                rows = m - i*nb;
                magma_queue_sync( queue[1] );
                lapackf77_cgetrf( &rows, &nb, work, &lda, ipiv+i*nb, &iinfo);
            }
            if (*info == 0 && iinfo > 0)
                *info = iinfo + i*nb;

            magma_cpermute_long2( n, dAT, dAT_offset, lddat, ipiv, nb, i*nb, queue[0] );

            // upload i-th panel
            magma_csetmatrix_async( m-i*nb, nb, work, 0, lda, dAP, 0, maxm,
                                    queue[1], NULL);
            magma_queue_sync( queue[1] );

            magma_ctranspose( dAT(i,i), lddat, dAP, 0, maxm, cols, nb, queue[0]);

            // do the small non-parallel computations
            if (s > (i+1)){
                magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dAT(i, i  ), lddat,
                                    dAT(i, i+1), lddat, queue[0]);
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             nb, m-(i+1)*nb, nb,
                             c_neg_one, dAT(i,   i+1), lddat,
                                        dAT(i+1, i  ), lddat,
                             c_one,     dAT(i+1, i+1), lddat, queue[0] );
            }
            else{
                magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-s*nb, nb,
                             c_one, dAT(i, i  ), lddat,
                                    dAT(i, i+1), lddat, queue[0] );
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(i+1)*nb, m-(i+1)*nb, nb,
                             c_neg_one, dAT(i,   i+1), lddat,
                                        dAT(i+1, i  ), lddat,
                             c_one,     dAT(i+1, i+1), lddat, queue[0] );
            }
        }
        
        magma_int_t nb0 = min(m - s*nb, n - s*nb);
        if ( nb0 > 0 ) {
            rows = m - s*nb;
            cols = maxm - s*nb;
    
            magma_ctranspose2( dAP, 0, maxm, dAT(s,s), lddat, nb0, rows, queue[0]);
            magma_queue_sync(queue[0]);
            magma_cgetmatrix_async( rows, nb0, dAP, 0, maxm, work, 0, lda, queue[1], NULL );
            magma_queue_sync(queue[1]);

            // do the cpu part
            lapackf77_cgetrf( &rows, &nb0, work, &lda, ipiv+s*nb, &iinfo);
            if (*info == 0 && iinfo > 0)
                *info = iinfo + s*nb;
            magma_cpermute_long2( n, dAT, dAT_offset, lddat, ipiv, nb0, s*nb, queue[0] );
    
            magma_csetmatrix_async( rows, nb0, work, 0, lda, dAP, 0, maxm, queue[1], NULL );
            magma_queue_sync(queue[1]);
            magma_ctranspose2( dAT(s,s), lddat, dAP, 0, maxm, rows, nb0, queue[0]);
    
            magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-s*nb-nb0, nb0,
                         c_one, dAT(s, s),     lddat,
                                dAT(s, s)+nb0, lddat, queue[0] );
        }
       
        if (maxdim*maxdim < 2*maxm*maxn) {
            magma_ctranspose_inplace(dAT, dAT_offset, lddat, ldda, queue[0] );
            magma_cgetmatrix( m, n, dA, dA_offset, ldda, a, 0, lda, queue[0] );
        } else {
            magma_ctranspose2( dA, dA_offset, ldda, dAT, dAT_offset, lddat, n, m, queue[0] );
            magma_cgetmatrix( m, n, dA, dA_offset, ldda, a, 0, lda, queue[0] );
            magma_queue_sync(queue[0]);
            magma_free( dAT );
        }

        magma_queue_sync(queue[0]);
        magma_free( dwork );
    }
    
    return *info;
} /* magma_cgetrf */

#undef dAT
