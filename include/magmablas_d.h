/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 * @generated from magmablas_z.h normal z -> d, Fri Jan 10 15:51:16 2014
 */

#ifndef MAGMA_BLAS_D_H
#define MAGMA_BLAS_D_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// copying sub-matrices (contiguous columns)
magma_err_t
magma_dsetmatrix(
    magma_int_t m, magma_int_t n,
    double const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaDouble_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue );

magma_err_t
magma_dsetvector(
    magma_int_t n,
    double const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaDouble_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_dgetmatrix(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    double*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue );

magma_err_t
magma_dgetvector(
    magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    double*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_dsetvector_async(
    magma_int_t n,
    double const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaDouble_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_dgetvector_async(
    magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    double*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_dsetmatrix_async(
    magma_int_t m, magma_int_t n,
    double const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaDouble_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_dgetmatrix_async(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    double*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue, magma_event_t *event );

magma_err_t
magma_dcopymatrix(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaDouble_ptr    dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue );

void dzero_nbxnb_block(
    int nb, magmaDouble_ptr dA, size_t dA_offset, int ldda,
    magma_queue_t queue );

void magmablas_dlaset(
    int uplo, magma_int_t m, magma_int_t n,
    magmaDouble_ptr A, size_t A_offset, magma_int_t lda,
    magma_queue_t queue );

void magmablas_dlacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaDouble_ptr dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

void magmablas_dswap(
    magma_int_t n,
    magmaDouble_ptr dA1T, size_t offset_dA1T, magma_int_t lda1,
    magmaDouble_ptr dA2T, size_t offset_dA2T, magma_int_t lda2,
    magma_queue_t queue );

void magmablas_dsetmatrix_1D_bcyclic( 
    magma_int_t m, magma_int_t n,
    const double *hA, magma_int_t lda, 
    magmaDouble_ptr *dA, magma_int_t ldda, 
    magma_int_t num_gpus, magma_int_t nb, 
    magma_queue_t* trans_queues);

void magmablas_dgetmatrix_1D_bcyclic( 
    magma_int_t m, magma_int_t n, 
    magmaDouble_ptr *dA, magma_int_t ldda, 
    double *hA, magma_int_t lda, 
    magma_int_t num_gpus, magma_int_t nb, 
    magma_queue_t* trans_queues);
// ========================================
// matrix transpose and swapping functions
magma_err_t
magma_dtranspose_inplace(
    magmaDouble_ptr dA, size_t dA_offset, int lda, int n,
    magma_queue_t queue );

magma_err_t
magma_dtranspose2(
    magmaDouble_ptr odata, size_t odata_offset, int ldo,
    magmaDouble_ptr idata, size_t idata_offset, int ldi,
    int m, int n,
    magma_queue_t queue );

magma_err_t
magma_dtranspose(
    magmaDouble_ptr odata, int offo, int ldo,
    magmaDouble_ptr idata, int offi, int ldi,
    int m, int n,
    magma_queue_t queue );

magma_err_t
magma_dpermute_long2(
    int n,
    magmaDouble_ptr dAT, size_t dAT_offset, int lda,
    int *ipiv, int nb, int ind,
    magma_queue_t queue );

magma_err_t 
magma_dpermute_long3(
    int n, 
    magmaDouble_ptr dAT, size_t dAT_offset, int lda, 
    int *ipiv, int nb, int ind, 
    magma_queue_t queue );

// ========================================
// BLAS functions
magma_err_t
magma_dgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    double beta,  magmaDouble_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_dgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dx, size_t dx_offset, magma_int_t incx,
    double beta,  magmaDouble_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    double beta,  magmaDouble_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_dsymv(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dx, size_t dx_offset, magma_int_t incx,
    double beta,  magmaDouble_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue );

magma_err_t
magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
    double beta,  magmaDouble_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

magma_err_t
magma_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

magma_err_t
magma_dtrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaDouble_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue );

magma_err_t
magma_dtrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue );

magma_err_t
magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaDouble_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    double beta, magmaDouble_ptr dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue );

// iwocl 2013 benchmark
void 
magmablas_dempty( magma_queue_t queue, magmaDouble_ptr dA, magmaDouble_ptr dB, magmaDouble_ptr dC);



#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_BLAS_H
