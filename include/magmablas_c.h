/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 * @generated from magmablas_z.h normal z -> c, Fri Jan 10 15:51:16 2014
 */

#ifndef MAGMA_BLAS_C_H
#define MAGMA_BLAS_C_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// copying sub-matrices (contiguous columns)
magma_err_t
magma_csetmatrix(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaFloatComplex_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue );

magma_err_t
magma_csetvector(
    magma_int_t n,
    magmaFloatComplex const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaFloatComplex_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_cgetmatrix(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue );

magma_err_t
magma_cgetvector(
    magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    magmaFloatComplex*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_csetvector_async(
    magma_int_t n,
    magmaFloatComplex const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaFloatComplex_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_cgetvector_async(
    magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    magmaFloatComplex*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_csetmatrix_async(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaFloatComplex_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_cgetmatrix_async(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_ccopymatrix(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex_ptr    dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue );

void czero_nbxnb_block(
    int nb, magmaFloatComplex_ptr dA, size_t dA_offset, int ldda,
    magma_queue_t queue );

void magmablas_claset(
    int uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr A, size_t A_offset, magma_int_t lda,
    magma_queue_t queue );

void magmablas_clacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaFloatComplex_ptr dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

void magmablas_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dA1T, size_t offset_dA1T, magma_int_t lda1,
    magmaFloatComplex_ptr dA2T, size_t offset_dA2T, magma_int_t lda2,
    magma_queue_t queue );

void magmablas_csetmatrix_1D_bcyclic( 
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA, magma_int_t lda, 
    magmaFloatComplex_ptr *dA, magma_int_t ldda, 
    magma_int_t num_gpus, magma_int_t nb, 
    magma_queue_t* trans_queues);

void magmablas_cgetmatrix_1D_bcyclic( 
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex_ptr *dA, magma_int_t ldda, 
    magmaFloatComplex *hA, magma_int_t lda, 
    magma_int_t num_gpus, magma_int_t nb, 
    magma_queue_t* trans_queues);
// ========================================
// matrix transpose and swapping functions
magma_err_t
magma_ctranspose_inplace(
    magmaFloatComplex_ptr dA, size_t dA_offset, int lda, int n,
    magma_queue_t queue );

magma_err_t
magma_ctranspose2(
    magmaFloatComplex_ptr odata, size_t odata_offset, int ldo,
    magmaFloatComplex_ptr idata, size_t idata_offset, int ldi,
    int m, int n,
    magma_queue_t queue );

magma_err_t
magma_ctranspose(
    magmaFloatComplex_ptr odata, int offo, int ldo,
    magmaFloatComplex_ptr idata, int offi, int ldi,
    int m, int n,
    magma_queue_t queue );

magma_err_t
magma_cpermute_long2(
    int n,
    magmaFloatComplex_ptr dAT, size_t dAT_offset, int lda,
    int *ipiv, int nb, int ind,
    magma_queue_t queue );

magma_err_t 
magma_cpermute_long3(
    int n, 
    magmaFloatComplex_ptr dAT, size_t dAT_offset, int lda, 
    int *ipiv, int nb, int ind, 
    magma_queue_t queue );

// ========================================
// BLAS functions
magma_err_t
magma_cgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha, magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloatComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    magmaFloatComplex beta,  magmaFloatComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_cgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha, magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloatComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloatComplex beta,  magmaFloatComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_chemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha, magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloatComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    magmaFloatComplex beta,  magmaFloatComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_chemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha, magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloatComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
    magmaFloatComplex beta,  magmaFloatComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_cherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha, magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
    float beta,  magmaFloatComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_ctrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha, magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloatComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

magma_err_t
magma_ctrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaFloatComplex_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue );

magma_err_t
magma_ctrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha, magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloatComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

magma_err_t
magma_cher2k(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha, magmaFloatComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaFloatComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    float beta, magmaFloatComplex_ptr dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

// iwocl 2013 benchmark
void 
magmablas_cempty( magma_queue_t queue, magmaFloat_ptr dA, magmaFloat_ptr dB, magmaFloat_ptr dC);



#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_BLAS_H
