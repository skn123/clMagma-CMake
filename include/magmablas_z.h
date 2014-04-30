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

#ifndef MAGMA_BLAS_Z_H
#define MAGMA_BLAS_Z_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// copying sub-matrices (contiguous columns)
magma_err_t
magma_zsetmatrix(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaDoubleComplex_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue );

magma_err_t
magma_zsetvector(
    magma_int_t n,
    magmaDoubleComplex const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaDoubleComplex_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_zgetmatrix(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue );

magma_err_t
magma_zgetvector(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    magmaDoubleComplex*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_zsetvector_async(
    magma_int_t n,
    magmaDoubleComplex const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaDoubleComplex_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_zgetvector_async(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    magmaDoubleComplex*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_zsetmatrix_async(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaDoubleComplex_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_zgetmatrix_async(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_zcopymatrix(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_ptr    dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue );

void zzero_nbxnb_block(
    int nb, magmaDoubleComplex_ptr dA, size_t dA_offset, int ldda,
    magma_queue_t queue );

void magmablas_zlaset(
    int uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr A, size_t A_offset, magma_int_t lda,
    magma_queue_t queue );

void magmablas_zlacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaDoubleComplex_ptr dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

void magmablas_zswap(
    magma_int_t n,
    magmaDoubleComplex_ptr dA1T, size_t offset_dA1T, magma_int_t lda1,
    magmaDoubleComplex_ptr dA2T, size_t offset_dA2T, magma_int_t lda2,
    magma_queue_t queue );

void magmablas_zsetmatrix_1D_bcyclic( 
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA, magma_int_t lda, 
    magmaDoubleComplex_ptr *dA, magma_int_t ldda, 
    magma_int_t num_gpus, magma_int_t nb, 
    magma_queue_t* trans_queues);

void magmablas_zgetmatrix_1D_bcyclic( 
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex_ptr *dA, magma_int_t ldda, 
    magmaDoubleComplex *hA, magma_int_t lda, 
    magma_int_t num_gpus, magma_int_t nb, 
    magma_queue_t* trans_queues);
// ========================================
// matrix transpose and swapping functions
magma_err_t
magma_ztranspose_inplace(
    magmaDoubleComplex_ptr dA, size_t dA_offset, int lda, int n,
    magma_queue_t queue );

magma_err_t
magma_ztranspose2(
    magmaDoubleComplex_ptr odata, size_t odata_offset, int ldo,
    magmaDoubleComplex_ptr idata, size_t idata_offset, int ldi,
    int m, int n,
    magma_queue_t queue );

magma_err_t
magma_ztranspose(
    magmaDoubleComplex_ptr odata, int offo, int ldo,
    magmaDoubleComplex_ptr idata, int offi, int ldi,
    int m, int n,
    magma_queue_t queue );

magma_err_t
magma_zpermute_long2(
    int n,
    magmaDoubleComplex_ptr dAT, size_t dAT_offset, int lda,
    int *ipiv, int nb, int ind,
    magma_queue_t queue );

magma_err_t 
magma_zpermute_long3(
    int n, 
    magmaDoubleComplex_ptr dAT, size_t dAT_offset, int lda, 
    int *ipiv, int nb, int ind, 
    magma_queue_t queue );

// ========================================
// BLAS functions
magma_err_t
magma_zgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_zgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_zhemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_zhemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_zherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
    double beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

magma_err_t
magma_ztrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaDoubleComplex_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue );

magma_err_t
magma_ztrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

magma_err_t
magma_zher2k(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    double beta, magmaDoubleComplex_ptr dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

// iwocl 2013 benchmark
void 
magmablas_zempty( magma_queue_t queue, magmaDouble_ptr dA, magmaDouble_ptr dB, magmaDouble_ptr dC);



#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_BLAS_H
