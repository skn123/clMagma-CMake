/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 * @precisions normal z -> s d c
 */

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "magmablas_z_trace.h"

#define PRECISION_z
#ifdef HAVE_clAmdBlas
#if defined(PRECISION_z) || defined(PRECISION_c)
#define clAmdBlasZhemvEx  clAmdBlasZhemv
#define clAmdBlasZherkEx  clAmdBlasZherk
#define clAmdBlasZher2kEx  clAmdBlasZher2k
#endif

// ========================================
// globals, defined in interface.c
extern cl_platform_id gPlatform;
extern cl_context     gContext;


magma_err_t
magma_zgemm_trace(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                                  magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
        magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue, magma_event_t* event )
{
    cl_int err = clAmdBlasZgemmEx(
            clAmdBlasColumnMajor,
            amdblas_trans_const( transA ),
            amdblas_trans_const( transB ),
            m, n, k,
            alpha, dA, dA_offset, lda,
            dB, dB_offset, ldb,
            beta,  dC, dC_offset, ldc,
            1, &queue, 0, NULL, event );
    clFlush(queue);
    return err;
}

magma_err_t
magma_zherk_trace(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t n, magma_int_t k,
        double alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        double beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue, magma_event_t* event )
{
    cl_int err = clAmdBlasZherkEx(
            clAmdBlasColumnMajor,
            amdblas_uplo_const( uplo ),
            amdblas_trans_const( trans ),
            n, k,
            alpha, dA, dA_offset, lda,
            beta,  dC, dC_offset, ldc,
            1, &queue, 0, NULL, event );
    clFlush(queue);
    return err;
}

magma_err_t
magma_ztrsm_trace(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
        magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, 
        magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
        magma_queue_t queue, magma_event_t* event )
{
    cl_int err = clAmdBlasZtrsmEx(
            clAmdBlasColumnMajor,
            amdblas_side_const( side ),
            amdblas_uplo_const( uplo ),
            amdblas_trans_const( trans ),
            amdblas_diag_const( diag ),
            m, n,
            alpha, dA, dA_offset, lda,
            dB, dB_offset, ldb,
            1, &queue, 0, NULL, event );
    clFlush(queue);
    return err;
}

#endif // HAVE_clAmdBlas

