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
// produces pointer and offset as two args to magmaBLAS routines
#define dA(i,j)  dA, ( (dA_offset) + (i) + (j)*ldda )

// produces pointer as single arg to BLAS routines
#define A(i,j)  &A[ (i) + (j)*lda ]

extern cl_context gContext;

magma_err_t
magma_zpotrf_gpu( magma_uplo_t   uplo, magma_int_t    n,
          magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
          magma_err_t*   info, magma_queue_t queue )
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

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)   
            On entry, the Hermitian matrix dA.  If UPLO = 'U', the leading   
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

// create 2nd queue
/*
    magma_queue_t queue2;
    magma_device_t device;
    int num = 0;
    magma_get_devices(&device, 1, &num);
    magma_queue_create(device, &queue2);
*/
// end
    
    magma_int_t j, jb, nb;
    magmaDoubleComplex  z_one = MAGMA_Z_MAKE(  1.0, 0.0 );
    magmaDoubleComplex mz_one = MAGMA_Z_MAKE( -1.0, 0.0 );
    double    one =  1.0;
    double  m_one = -1.0;
    magmaDoubleComplex* work;
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
    
    nb = magma_get_zpotrf_nb( n );
    
    err = magma_zmalloc_cpu(  &work, nb*nb );
    if ( err != MAGMA_SUCCESS ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    magma_event_t* event;
    //magma_event_t event1 = NULL;
    
    if ((nb <= 1) || (nb >= n)) {
        // use unblocked code
        chk( magma_zgetmatrix( n, n, dA, dA_offset, ldda, work, 0, n, queue ));
        lapackf77_zpotrf( lapack_uplo_const(uplo), &n, work, &n, info );
        chk( magma_zsetmatrix( n, n, work, 0, n, dA, dA_offset, ldda, queue ));
    }
    else {
        //magma_queue_t queues[2] = {queue, queue2};    
        trace_init(1, 1, 1, &queue);
        //trace_init(1, 1, 2, queues);
        magma_event_t* trace_event;
        trace_cpu_start(0, "dummy", "dummy");
        trace_cpu_end(0);
        /*
        trace_gpu_start(0, 0, "dummy", "dummy");
        cl_int err;
        trace_event = trace_gpu_end(0, 0);
        *trace_event = clCreateUserEvent(gContext, &err);
        clSetUserEventStatus(*trace_event, CL_COMPLETE);
        */
        if ( uplo == MagmaUpper ) {
            // --------------------
            // compute Cholesky factorization A = U'*U
            // using the left looking algorithm
            for( j = 0; j < n; j += nb ) {
                // apply all previous updates to diagonal block
                jb = min( nb, n-j );
                if ( j > 0 ) {
                    trace_gpu_start(0, 0, "herk", "herk");
                    trace_event = trace_gpu_end(0, 0);
                    chk( magma_zherk_trace( MagmaUpper, MagmaConjTrans, jb, j,
                        m_one, dA(0,j), ldda,
                          one, dA(j,j), ldda, queue, trace_event ));
                    //clFinish(queue);
                }
                
                // start asynchronous data transfer
                
                trace_gpu_start(0, 0, "comm_get", "get");
                trace_event = trace_gpu_end(0, 0);
                event = trace_event;
                
                //trace_cpu_start(0, "comm_get", "get");
                //chk( magma_zgetmatrix_async( jb, jb, dA(j,j), ldda, work, 0, jb, queue, &event1 ));
                chk( magma_zgetmatrix_async( jb, jb, dA(j,j), ldda, work, 0, jb, queue, event ));

                // apply all previous updates to block row right of diagonal block
                if ( j+jb < n && j > 0) {
                    trace_gpu_start(0, 0, "gemm", "gemm");
                    trace_event = trace_gpu_end(0, 0);
                    chk( magma_zgemm_trace( MagmaConjTrans, MagmaNoTrans,
                        jb, n-j-jb, j,
                        mz_one, dA(0, j   ), ldda,
                                dA(0, j+jb), ldda,
                        z_one,  dA(j, j+jb), ldda, queue, trace_event ));
                }
                
                // simultaneous with above zgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                chk( magma_event_sync( *event ));
                //chk( magma_event_sync( event1 ));
                trace_cpu_start(0, "potrf", "potrf");
                lapackf77_zpotrf( MagmaUpperStr, &jb, work, &jb, info );
                trace_cpu_end(0);
                if ( *info != 0 ) {
                    assert( *info > 0 );
                    *info += j;
                    break;
                }
                
                trace_gpu_start(0, 0, "comm_set", "set");
                trace_event = trace_gpu_end(0, 0);
                event = trace_event;
                
                //trace_cpu_start(0, "comm_set", "set");
                //chk( magma_zsetmatrix_async( jb, jb, work, 0, jb, dA(j,j), ldda, queue, &event1 ));
                chk( magma_zsetmatrix_async( jb, jb, work, 0, jb, dA(j,j), ldda, queue, trace_event ));
                
                // apply diagonal block to block row right of diagonal block
                if ( j+jb < n ) {
                    chk( magma_event_sync( *event ));
                    //chk( magma_event_sync( event1 ));
    //                trace_cpu_end(0);
                    trace_gpu_start(0, 0, "trsm", "trsm");
                    trace_event = trace_gpu_end(0, 0);
                    chk( magma_ztrsm_trace(
                        MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                        jb, n-j-jb,
                        z_one, dA(j, j),    ldda,
                               dA(j, j+jb), ldda, queue, trace_event ));
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
                if(j>0){
                    trace_gpu_start(0, 0, "herk", "herk");
                    trace_event = trace_gpu_end(0, 0);
                    chk( magma_zherk_trace( MagmaLower, MagmaNoTrans, jb, j,
                        m_one, dA(j, 0), ldda,
                        one, dA(j, j), ldda, queue, trace_event ));
                    //clFinish(queue);
                }
                // start asynchronous data transfer
                
                trace_gpu_start(0, 0, "comm_get", "get");
                trace_event = trace_gpu_end(0, 0);
                event = trace_event;
                
                //trace_cpu_start(0, "comm_get", "get");
                //chk( magma_zgetmatrix_async( jb, jb, dA(j,j), ldda, work, 0, jb, queue, &event1 ));
                chk( magma_zgetmatrix_async( jb, jb, dA(j,j), ldda, work, 0, jb, queue, event ));
                //chk( magma_event_sync( *event ));

                // apply all previous updates to block column below diagonal block
                if ( j+jb < n && j > 0) {
                    trace_gpu_start(0, 0, "gemm", "gemm");
                    trace_event = trace_gpu_end(0, 0);
                    chk( magma_zgemm_trace( MagmaNoTrans, MagmaConjTrans,
                        n-j-jb, jb, j,
                        mz_one, dA(j+jb, 0), ldda,
                                dA(j,    0), ldda,
                        z_one,  dA(j+jb, j), ldda, queue, trace_event ));
                    /*
                    chk( magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                        n-j-jb, jb, j,
                        mz_one, dA(j+jb, 0), ldda,
                                dA(j,    0), ldda,
                        z_one,  dA(j+jb, j), ldda, queue));
                    */
                }
                
                // simultaneous with above zgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                /*
                cl_int status;
                size_t return_size;
                clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, &return_size);
                if(status != CL_COMPLETE){
                    //printf("%d\n", status);
                */
                chk( magma_event_sync( *event ));
                //}
                //chk( magma_event_sync( event1 ));
                trace_cpu_start(0, "potrf", "potrf");
                lapackf77_zpotrf( MagmaLowerStr, &jb, work, &jb, info );
                trace_cpu_end(0);
                if ( *info != 0 ) {
                    assert( *info > 0 );
                    *info += j;
                    break;
                }
                
                trace_gpu_start(0, 0, "comm_set", "set");
                trace_event = trace_gpu_end(0, 0);
                event = trace_event;
                
                //trace_cpu_start(0, "comm_set", "set");
                //chk( magma_zsetmatrix_async( jb, jb, work, 0, jb, dA(j,j), ldda, queue, &event1 ));
                chk( magma_zsetmatrix_async( jb, jb, work, 0, jb, dA(j,j), ldda, queue, trace_event ));
                
                // apply diagonal block to block column below diagonal
                if ( j+jb < n ) {
                    chk( magma_event_sync( *event ));
                    //chk( magma_event_sync( event1 ));
    //                trace_cpu_end(0);
                    trace_gpu_start(0, 0, "trsm", "trsm");
                    trace_event = trace_gpu_end(0, 0);
                    chk( magma_ztrsm_trace(
                        MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                        n-j-jb, jb,
                        z_one, dA(j, j   ), ldda,
                               dA(j+jb, j), ldda, queue, trace_event ));
                }
            }
        }
    
    }
    chk( magma_queue_sync( queue ));
//    chk( magma_queue_sync( queue2 ));
//    magma_queue_destroy( queue2 );
    trace_finalize("zpotrf.svg", "trace.css");
    chk( magma_free_cpu( work ));
    
    return *info;
}
