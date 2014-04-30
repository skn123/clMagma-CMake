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
#ifndef MAGMA_BLAS_Z_TRACE_H
#define MAGMA_BLAS_Z_TRACE_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

magma_err_t
magma_zgemm_trace(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
        magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue, magma_event_t* event );

magma_err_t
magma_zherk_trace(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t n, magma_int_t k,
        double alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        double beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue, magma_event_t* event );

magma_err_t
magma_ztrsm_trace(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
        magma_queue_t queue, magma_event_t* event );

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_BLAS_H
