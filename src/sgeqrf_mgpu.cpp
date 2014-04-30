/*
   -- clMAGMA (version 1.1.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date January 2014

   @generated from zgeqrf_mgpu.cpp normal z -> s, Fri Jan 10 15:51:18 2014

 */
#include "common_magma.h"

extern "C" magma_err_t
magma_sgeqrf2_mgpu( magma_int_t num_gpus, magma_int_t m, magma_int_t n,
        magmaFloat_ptr *dlA, magma_int_t ldda,
        float *tau, 
        magma_int_t *info, 
        magma_queue_t *queues)
{
    /*  -- clMAGMA (version 1.1.0) --
        Univ. of Tennessee, Knoxville
        Univ. of California, Berkeley
        Univ. of Colorado, Denver
        @date January 2014

        Purpose
        =======
        SGEQRF2_MGPU computes a QR factorization of a real M-by-N matrix A:
        A = Q * R. This is a GPU interface of the routine.

        Arguments
        =========
        M       (input) INTEGER
        The number of rows of the matrix A.  M >= 0.

        N       (input) INTEGER
        The number of columns of the matrix A.  N >= 0.

        dA      (input/output) REAL array on the GPU, dimension (LDDA,N)
        On entry, the M-by-N matrix dA.
        On exit, the elements on and above the diagonal of the array
        contain the min(M,N)-by-N upper trapezoidal matrix R (R is
        upper triangular if m >= n); the elements below the diagonal,
        with the array TAU, represent the orthogonal matrix Q as a
        product of min(m,n) elementary reflectors (see Further
        Details).

        LDDA    (input) INTEGER
        The leading dimension of the array dA.  LDDA >= max(1,M).
        To benefit from coalescent memory accesses LDDA must be
        dividable by 16.

        TAU     (output) REAL array, dimension (min(M,N))
        The scalar factors of the elementary reflectors (see Further
        Details).

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

        where tau is a real scalar, and v is a real vector with
        v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
        and tau in TAU(i).
        =====================================================================    */

#define dlA(gpu,a_1,a_2) dlA[gpu], ((a_2)*(ldda) + (a_1))
#define dlA_offset(a_1, a_2) ((a_2)*(ldda) + (a_1))
#define work_ref(a_1)    ( work + (a_1))
#define hwork            ( work + (nb)*(m))

#define hwrk_ref(a_1)    ( local_work + (a_1))
#define lhwrk            ( local_work + (nb)*(m))

    magmaFloat_ptr dwork[4], panel[4];
    size_t panel_offset[4];
    float *local_work;

    magma_int_t i, j, k, ldwork, lddwork, old_i, old_ib, rows;
    magma_int_t nbmin, nx, ib, nb;
    magma_int_t lhwork, lwork;

    int panel_gpunum, i_local, n_local[4], la_gpu, displacement; 

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    k = min(m,n);
    if (k == 0)
        return *info;

    nb = magma_get_sgeqrf_nb(m);

    displacement = n * nb;
    lwork  = (m+n+64) * nb;
    lhwork = lwork - (m)*nb;

    for(i=0; i<num_gpus; i++){
        if (MAGMA_SUCCESS != magma_smalloc( &(dwork[i]), (n + ldda)*nb )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
    }

    /* Set the number of local n for each GPU */
    for(i=0; i<num_gpus; i++){
        n_local[i] = ((n/nb)/num_gpus)*nb;
        if (i < (n/nb)%num_gpus)
            n_local[i] += nb;
        else if (i == (n/nb)%num_gpus)
            n_local[i] += n%nb;
    }

    if (MAGMA_SUCCESS != magma_smalloc_cpu( (&local_work), lwork )) {
        *info = -9;
        for(i=0; i<num_gpus; i++){
            magma_free( dwork[i] );
        }

        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    nbmin = 2;
    nx    = nb;
    ldwork = m;
    lddwork= n;

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code initially */
        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nx; i += nb) 
        {
            /* Set the GPU number that holds the current panel */
            panel_gpunum = (i/nb)%num_gpus;

            /* Set the local index where the current panel is */
            i_local = i/(nb*num_gpus)*nb;

            ib = min(k-i, nb);
            rows = m -i;
            /* Send current panel to the CPU */
            magma_queue_sync(queues[panel_gpunum*2]);
            magma_sgetmatrix_async( rows, ib,
                    dlA(panel_gpunum, i, i_local), ldda,
                    hwrk_ref(i), 0, ldwork, queues[panel_gpunum*2+1], NULL );

            if (i>0){
                /* Apply H' to A(i:m,i+2*ib:n) from the left; this is the look-ahead
                   application to the trailing matrix                                     */
                la_gpu = panel_gpunum;

                /* only the GPU that has next panel is done look-ahead */
                magma_slarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
                        m-old_i, n_local[la_gpu]-i_local-old_ib, old_ib,
                        panel[la_gpu], panel_offset[la_gpu], ldda, 
                        dwork[la_gpu], 0, lddwork,
                        dlA(la_gpu, old_i, i_local+old_ib), ldda, 
                        dwork[la_gpu], old_ib, lddwork, queues[2*la_gpu]);

                la_gpu = ((i-nb)/nb)%num_gpus;
                magma_ssetmatrix_async( old_ib, old_ib,
                        hwrk_ref(old_i), 0, ldwork,
                        panel[la_gpu], panel_offset[la_gpu], ldda, queues[la_gpu*2], NULL );
            }

            magma_queue_sync( queues[panel_gpunum*2+1] );

            lapackf77_sgeqrf(&rows, &ib, hwrk_ref(i), &ldwork, tau+i, lhwrk, &lhwork, info);

            // Form the triangular factor of the block reflector
            // H = H(i) H(i+1) . . . H(i+ib-1) 
            lapackf77_slarft( MagmaForwardStr, MagmaColumnwiseStr,
                    &rows, &ib,
                    hwrk_ref(i), &ldwork, tau+i, lhwrk, &ib);

            spanel_to_q( MagmaUpper, ib, hwrk_ref(i), ldwork, lhwrk+ib*ib );
            // Send the current panel back to the GPUs 
            // Has to be done with asynchronous copies

            for(j=0; j<num_gpus; j++)
            {  
                if (j == panel_gpunum){
                    panel[j] = dlA(j, i, i_local);
                    panel_offset[j] = dlA_offset(i, i_local);
                }
                else{
                    panel[j] = dwork[j];
                    panel_offset[j] = displacement;
                }
                magma_queue_sync( queues[j*2] );
                magma_ssetmatrix_async( rows, ib,
                        hwrk_ref(i), 0, ldwork,
                        panel[j], panel_offset[j], ldda, queues[j*2+1], NULL );

                /* Send the T matrix to the GPU. 
                   Has to be done with asynchronous copies */
                magma_ssetmatrix_async( ib, ib, lhwrk, 0, ib,
                                        dwork[j], 0, lddwork, queues[2*j+1], NULL );
            }

            for(j=0; j<num_gpus; j++)
            {
                magma_queue_sync( queues[j*2+1] );
            }

            if (i + ib < n) 
            {
                 if (i+nb < k-nx)
                {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left;
                       This is update for the next panel; part of the look-ahead    */
                    la_gpu = (panel_gpunum+1)%num_gpus;
                    int i_loc = (i+nb)/(nb*num_gpus)*nb;
                    for(j=0; j<num_gpus; j++){
                        if (j==la_gpu)
                            magma_slarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
                                    rows, ib, ib,
                                    panel[j], panel_offset[j], ldda, 
                                    dwork[j], 0, lddwork,
                                    dlA(j, i, i_loc), ldda, 
                                    dwork[j], ib, lddwork, 
                                    queues[j*2]);
                        else if (j<=panel_gpunum)
                            magma_slarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
                                    rows, n_local[j]-i_local-ib, ib,
                                    panel[j], panel_offset[j], ldda, 
                                    dwork[j], 0,   lddwork,
                                    dlA(j, i, i_local+ib), ldda, 
                                    dwork[j], ib, lddwork,
                                    queues[j*2]);
                        else
                            magma_slarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
                                    rows, n_local[j]-i_local, ib,
                                    panel[j], panel_offset[j], ldda, 
                                    dwork[j], 0, lddwork,
                                    dlA(j, i, i_local), ldda, 
                                    dwork[j], ib, lddwork, 
                                    queues[j*2]);
                    }

                    /* Restore the panel */
                    sq_to_panel( MagmaUpper, ib, hwrk_ref(i), ldwork, lhwrk+ib*ib );
                }
                else {
                    /* do the entire update as we exit and there would be no lookahead */
                    la_gpu = (panel_gpunum+1)%num_gpus;
                    int i_loc = (i+nb)/(nb*num_gpus)*nb;

                    magma_slarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
                            rows, n_local[la_gpu]-i_loc, ib,
                            panel[la_gpu], panel_offset[la_gpu], ldda, 
                            dwork[la_gpu], 0, lddwork,
                            dlA(la_gpu, i, i_loc), ldda, 
                            dwork[la_gpu], ib, lddwork,
                            queues[la_gpu*2]);
 
                    /* Restore the panel */
                    sq_to_panel( MagmaUpper, ib, hwrk_ref(i), ldwork, lhwrk+ib*ib ); 
                    
                    //magma_setdevice(panel_gpunum);                    
                    magma_ssetmatrix( ib, ib,
                            hwrk_ref(i), 0, ldwork,
                            dlA(panel_gpunum, i, i_local), ldda,
                            queues[panel_gpunum*2]);
                }
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    for(j=0; j<num_gpus; j++){
        magma_free( dwork[j] );
    }

    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib   = n-i;
        rows = m-i;
        lhwork = lwork - rows*ib;

        panel_gpunum = (panel_gpunum+1)%num_gpus;
        int i_loc = (i)/(nb*num_gpus)*nb;

        magma_sgetmatrix( rows, ib,
                dlA(panel_gpunum, i, i_loc), ldda,
                lhwrk, 0, rows, 
                queues[panel_gpunum*2]);

        lhwork = lwork - rows*ib;
        lapackf77_sgeqrf(&rows, &ib, lhwrk, &rows, tau+i, lhwrk+ib*rows, &lhwork, info);

        magma_ssetmatrix( rows, ib,
                lhwrk, 0, rows,
                dlA(panel_gpunum, i, i_loc), ldda, 
                queues[panel_gpunum*2]);
    }

    magma_free_cpu( local_work );

    return *info;
} /* magma_sgeqrf2_mgpu */
