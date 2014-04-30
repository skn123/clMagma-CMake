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

#define USE_PINNED_CLMEMORY
#ifdef  USE_PINNED_CLMEMORY
extern cl_context gContext;
#endif

#define inAT(id,i,j)  d_lAT[(id)], (((i)*nb)*lddat + (j)*nb)
#define inA( id,i,j)  d_lA[(id)],  (((i)*nb)+ldda  * (j)*nb)

extern "C" magma_err_t
magma_zgetrf_msub(magma_int_t trans, magma_int_t num_subs, magma_int_t num_gpus, 
                 magma_int_t m, magma_int_t n, 
                 magmaDoubleComplex_ptr *d_lA, size_t dlA_offset, magma_int_t ldda,
                 magma_int_t *ipiv, magma_int_t *info,
                 magma_queue_t *queues)
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
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.
    =====================================================================    */


    magma_int_t maxm, tot_subs = num_subs*num_gpus;
    magma_int_t i, j, d, lddat;
    /* submatrix info */
    magma_int_t nb, n_local[ MagmaMaxSubs * MagmaMaxGPUs ];
    magmaDoubleComplex_ptr d_lAT[ MagmaMaxSubs * MagmaMaxGPUs ];
    /* local workspace per GPU */
    magmaDoubleComplex_ptr d_panel[ MagmaMaxGPUs ];
    magmaDoubleComplex_ptr d_lAP[ MagmaMaxGPUs ];
    magmaDoubleComplex *work;

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -2;
    else if (n < 0)
        *info = -3;
    else if (trans == MagmaTrans && ldda < max(1,m))
        *info = -5;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    nb = magma_get_zgetrf_nb(m);

    if (nb <= 1 || nb >= n) {
        /* Use CPU code. */
        magma_zmalloc_cpu( &work, m * n );
        if (work == NULL) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        magma_zgetmatrix( m, n, d_lA[0], 0, ldda, work, 0, m, queues[0] );
        lapackf77_zgetrf(&m, &n, work, &m, ipiv, info);
        magma_zsetmatrix( m, n, work, 0, m, d_lA[0], 0, ldda, queues[0] );
        magma_free_cpu(work);
    } else {
        /* Use hybrid blocked code. */
        maxm = ((m + 31)/32)*32;
        if (tot_subs > ceil((double)n/nb)) {
            printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) tot_subs );
            *info = -1;
            return *info;
        }

        /* allocate workspace for each GPU */
        lddat = n/nb;           /* number of block columns         */
        lddat = lddat/tot_subs; /* number of block columns per GPU */
        lddat = nb*lddat;       /* number of columns per GPU       */
        if (lddat * tot_subs < n) {
            /* left over */
            if (n-lddat*tot_subs >= nb) {
                lddat += nb;
            } else {
                lddat += (n-lddat*tot_subs)%nb;
            }
        }
        lddat = ((lddat+31)/32)*32; /* make it a multiple of 32 */
        /* allocating workspace */
        for (d=0; d<num_gpus; d++) {
            //#define SINGLE_GPU_PER_CONTEXT
            #ifdef SINGLE_GPU_PER_CONTEXT
            if ((MAGMA_SUCCESS != magma_zmalloc_mgpu( d, &d_panel[d], (2+num_gpus)*nb*maxm ))  ||
                (MAGMA_SUCCESS != magma_zmalloc_mgpu( d, &d_lAP[d],   (2+num_gpus)*nb*maxm )) ) {
            #else
            if ((MAGMA_SUCCESS != magma_zmalloc( &d_panel[d], (2+num_gpus)*nb*maxm ))  ||
                (MAGMA_SUCCESS != magma_zmalloc( &d_lAP[d], (2+num_gpus)*nb*maxm )) ) {
            #endif
                for( i=0; i<d; i++ ) {
                    magma_free( d_panel[i] );
                    magma_free( d_lAP[i] );
                }
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
        }
        /* transposing the local matrix */
        for (i=0; i<tot_subs; i++) {
            /* local-n and local-ld */
            n_local[i] = ((n/nb)/tot_subs)*nb;
            if (i < (n/nb)%tot_subs)
               n_local[i] += nb;
            else if (i == (n/nb)%tot_subs)
               n_local[i] += n%nb;

            /* local-matrix storage */
            if (trans == MagmaNoTrans) {
                d_lAT[i] = d_lA[i];
            } else {
                if ((m == n_local[i]) && (m%32 == 0) && (ldda%32 == 0) && (ldda == lddat)) {
                    d_lAT[i] = d_lA[i];
                    magma_ztranspose_inplace( d_lA[i], 0, ldda, ldda, queues[2*(i%num_gpus)+1] );
                } else {
                    #ifdef SINGLE_GPU_PER_CONTEXT
                    if (MAGMA_SUCCESS != magma_zmalloc_mgpu( i%num_gpus, &d_lAT[i], lddat*maxm )) {
                    #else
                    if (MAGMA_SUCCESS != magma_zmalloc( &d_lAT[i], lddat*maxm )) {
                    #endif
                        for (j=0; j<=i; j++) {
                            magma_free( d_panel[j] );
                            magma_free( d_lAP[j] );
                        }
                        for (j=0; j<i; j++) {
                            if (d_lAT[j] != d_lA[j]) magma_free( d_lAT[j] );
                        }
                        *info = MAGMA_ERR_DEVICE_ALLOC;
                        return *info;
                    }
                    magma_ztranspose2(d_lAT[i], 0, lddat, d_lA[i], 0, ldda, m, n_local[i], queues[2*(i%num_gpus)+1]);
                }
            }
        }
        if (trans == MagmaNoTrans) {
            for (d=0; d<num_gpus; d++){
                magma_queue_sync(queues[2*d+1]);
            }
        }

        /* cpu workspace */
        #ifdef USE_PINNED_CLMEMORY
        cl_mem buffer = clCreateBuffer(gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(magmaDoubleComplex)*maxm*nb*(1+num_gpus), NULL, NULL);
        for (d=0; d<num_gpus; d++) {
            work = (magmaDoubleComplex*)clEnqueueMapBuffer(queues[2*d], buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                                           sizeof(magmaDoubleComplex)*maxm*nb*(1+num_gpus), 0, NULL, NULL, NULL);
        }
        #else
        if (MAGMA_SUCCESS != magma_zmalloc_cpu( &work, maxm*nb*(1+num_gpus) )) {
            for(d=0; d<num_gpus; d++ ) magma_free( d_panel[d] );
            for(d=0; d<tot_subs; d++ ) {
                if( d_lAT[d] != d_lA[d] ) magma_free( d_lAT[d] );
            }
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        #endif

        /* calling multi-gpu interface with allocated workspaces and streams */
        magma_zgetrf2_msub(num_subs, num_gpus, m, n, nb, 0, d_lAT, 0, lddat, ipiv, d_lAP, d_panel, 0, work, maxm,
                           info, queues);

        /* save on output */
        for (d=0; d<tot_subs; d++) {
            if (trans == MagmaNoTrans) {
                //magma_zcopymatrix( n_local[d], m, d_lAT[d], 0, lddat, d_lA[d], 0, ldda, queues[2*d+1] );
            } else {
                if (d_lAT[d] == d_lA[d]) {
                    magma_ztranspose_inplace( d_lA[d], 0, ldda, ldda, queues[2*(d%num_gpus)+1] );
                } else {
                    magma_ztranspose2( d_lA[d], 0, ldda, d_lAT[d], 0, lddat, n_local[d], m, queues[2*(d%num_gpus)+1] );
                }
            }
        }
        /* clean up */
        for (d=0; d<num_gpus; d++) {
            magma_queue_sync(queues[2*d+1]);
            magma_free( d_panel[d] );
            magma_free( d_lAP[d] );
            d_panel[d] = d_lAP[d] = NULL;
        } 
        for (d=0; d<tot_subs; d++) {
            if (d_lAT[d] != d_lA[d]) {
                magma_free( d_lAT[d] ); 
                d_lAT[d] = NULL;
            }
        }
        #ifdef USE_PINNED_CLMEMORY
        for (d=0; d<num_gpus; d++) {
            clEnqueueUnmapMemObject(queues[2*d], buffer, work, 0, NULL, NULL);
        }
        clReleaseMemObject( buffer );
        #else
        magma_free_cpu( work );
        #endif
        work = NULL;
      }
      return *info;       
      /* End of MAGMA_ZGETRF_MSUB */
}

