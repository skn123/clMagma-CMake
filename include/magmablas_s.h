/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 * @generated from magmablas_z.h normal z -> s, Fri Jan 10 15:51:16 2014
 */

#ifndef MAGMA_BLAS_S_H
#define MAGMA_BLAS_S_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// copying sub-matrices (contiguous columns)
magma_err_t
magma_ssetmatrix(
    magma_int_t m, magma_int_t n,
    float const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaFloat_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue );

magma_err_t
magma_ssetvector(
    magma_int_t n,
    float const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaFloat_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_sgetmatrix(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    float*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue );

magma_err_t
magma_sgetvector(
    magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    float*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_ssetvector_async(
    magma_int_t n,
    float const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaFloat_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_sgetvector_async(
    magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    float*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_ssetmatrix_async(
    magma_int_t m, magma_int_t n,
    float const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaFloat_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_sgetmatrix_async(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    float*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_scopymatrix(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr    dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue );

void szero_nbxnb_block(
    int nb, magmaFloat_ptr dA, size_t dA_offset, int ldda,
    magma_queue_t queue );

void magmablas_slaset(
    int uplo, magma_int_t m, magma_int_t n,
    magmaFloat_ptr A, size_t A_offset, magma_int_t lda,
    magma_queue_t queue );

void magmablas_slacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaFloat_ptr dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

void magmablas_sswap(
    magma_int_t n,
    magmaFloat_ptr dA1T, size_t offset_dA1T, magma_int_t lda1,
    magmaFloat_ptr dA2T, size_t offset_dA2T, magma_int_t lda2,
    magma_queue_t queue );

void magmablas_ssetmatrix_1D_bcyclic( 
    magma_int_t m, magma_int_t n,
    const float *hA, magma_int_t lda, 
    magmaFloat_ptr *dA, magma_int_t ldda, 
    magma_int_t num_gpus, magma_int_t nb, 
    magma_queue_t* trans_queues);

void magmablas_sgetmatrix_1D_bcyclic( 
    magma_int_t m, magma_int_t n, 
    magmaFloat_ptr *dA, magma_int_t ldda, 
    float *hA, magma_int_t lda, 
    magma_int_t num_gpus, magma_int_t nb, 
    magma_queue_t* trans_queues);
// ========================================
// matrix transpose and swapping functions
magma_err_t
magma_stranspose_inplace(
    magmaFloat_ptr dA, size_t dA_offset, int lda, int n,
    magma_queue_t queue );

magma_err_t
magma_stranspose2(
    magmaFloat_ptr odata, size_t odata_offset, int ldo,
    magmaFloat_ptr idata, size_t idata_offset, int ldi,
    int m, int n,
    magma_queue_t queue );

magma_err_t
magma_stranspose(
    magmaFloat_ptr odata, int offo, int ldo,
    magmaFloat_ptr idata, int offi, int ldi,
    int m, int n,
    magma_queue_t queue );

magma_err_t
magma_spermute_long2(
    int n,
    magmaFloat_ptr dAT, size_t dAT_offset, int lda,
    int *ipiv, int nb, int ind,
    magma_queue_t queue );

magma_err_t 
magma_spermute_long3(
    int n, 
    magmaFloat_ptr dAT, size_t dAT_offset, int lda, 
    int *ipiv, int nb, int ind, 
    magma_queue_t queue );

// ========================================
// BLAS functions
magma_err_t
magma_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    float beta,  magmaFloat_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_sgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    float beta,  magmaFloat_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    float beta,  magmaFloat_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_ssymv(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    float beta,  magmaFloat_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
    float beta,  magmaFloat_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

magma_err_t
magma_strsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue );

magma_err_t
magma_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

magma_err_t
magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    float beta, magmaFloat_ptr dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

// iwocl 2013 benchmark
void 
magmablas_sempty( magma_queue_t queue, magmaFloat_ptr dA, magmaFloat_ptr dB, magmaFloat_ptr dC);



#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_BLAS_H
