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

#ifndef MAGMA_Z_H
#define MAGMA_Z_H

#include "magma_types.h"

#define PRECISION_z

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
magma_int_t magma_get_zpotrf_nb( magma_int_t m );
magma_int_t magma_get_zgetrf_nb( magma_int_t m );
magma_int_t magma_get_zgetri_nb( magma_int_t m );
magma_int_t magma_get_zgeqrf_nb( magma_int_t m );
magma_int_t magma_get_zgeqlf_nb( magma_int_t m );
magma_int_t magma_get_zgehrd_nb( magma_int_t m );
magma_int_t magma_get_zhetrd_nb( magma_int_t m );
magma_int_t magma_get_zgelqf_nb( magma_int_t m );
magma_int_t magma_get_zgebrd_nb( magma_int_t m );
magma_int_t magma_get_zhegst_nb( magma_int_t m );
magma_int_t magma_get_zgesvd_nb( magma_int_t m );

/* ////////////////////////////////////////////////////////////////////////////
    -- MAGMA function definitions / Data on CPU
*/

magma_err_t
magma_zgebrd(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *a, magma_int_t lda, double *d, double *e,
    magmaDoubleComplex *tauq, magmaDoubleComplex *taup,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_err_t 
magma_zgeqrf( 
    magma_int_t m, magma_int_t n, magmaDoubleComplex *A,
    magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work,
    magma_int_t lwork, magma_int_t *info, magma_queue_t *queue);

magma_err_t 
magma_zgesv ( 
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t 
magma_zgetrf( 
    magma_int_t m, magma_int_t n, magmaDoubleComplex *A,
    magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t *queue );

magma_err_t
magma_zposv (
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t
magma_zpotrf(
    magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A,
    magma_int_t lda, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t
magma_zunghr(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT, size_t dT_offset,
    magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_zungqr(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT, size_t dT_offset,
    magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zunmtr(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *a,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *c,    magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zunmqr(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *a,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *c,    magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zunmql(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *c, magma_int_t ldc,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

#if defined(PRECISION_z) || defined(PRECISION_c)

magma_int_t
magma_zgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *geev_w_array,
    magmaDoubleComplex *vl, magma_int_t ldvl,
    magmaDoubleComplex *vr, magma_int_t ldvr,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zgesvd(
    char jobu, char jobvt, magma_int_t m, magma_int_t n,
    magmaDoubleComplex *a,    magma_int_t lda, double *s,
    magmaDoubleComplex *u,    magma_int_t ldu,
    magmaDoubleComplex *vt,   magma_int_t ldvt,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zheevd(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *a, magma_int_t lda,
    double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zstedx(
    magma_vec_t range, magma_int_t n, double vl, double vu,
    magma_int_t il, magma_int_t iu, double* d, double* e,
    magmaDoubleComplex* z, magma_int_t ldz,
    double* rwork, magma_int_t lrwork,
    magma_int_t* iwork, magma_int_t liwork,
    magmaDouble_ptr dwork,
    magma_int_t* info, magma_queue_t queue );

#else

magma_int_t
magma_zgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *WR, double *WI,
    magmaDoubleComplex *vl, magma_int_t ldvl,
    magmaDoubleComplex *vr, magma_int_t ldvr,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zgesvd(
    char jobu, char jobvt, magma_int_t m, magma_int_t n,
    magmaDoubleComplex *a,    magma_int_t lda, double *s,
    magmaDoubleComplex *u,    magma_int_t ldu,
    magmaDoubleComplex *vt,   magma_int_t ldvt,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zheevd(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex *a, magma_int_t lda,
    double *w,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zstedx(
    magma_vec_t range, magma_int_t n, double vl, double vu,
    magma_int_t il, magma_int_t iu, double* d, double* e,
    double* z, magma_int_t ldz,
    double* rwork, magma_int_t lrwork,
    magma_int_t* iwork, magma_int_t liwork,
    magmaDouble_ptr dwork,
    magma_int_t* info, magma_queue_t queue );

magma_int_t
magma_zlaex0(
    magma_int_t n, double* d, double* e, double* q, magma_int_t ldq,
    double* work, magma_int_t* iwork, magmaDouble_ptr dwork,
    magma_vec_t range, double vl, double vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t* info, magma_queue_t queue );

magma_int_t
magma_zlaex1(
    magma_int_t n, double* d, double* q, magma_int_t ldq,
    magma_int_t* indxq, double rho, magma_int_t cutpnt,
    double* work, magma_int_t* iwork, magmaDouble_ptr dwork,
    magma_vec_t range, double vl, double vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t* info, magma_queue_t queue );


magma_int_t
magma_zlaex3(
    magma_int_t k, magma_int_t n, magma_int_t n1, double* d,
    double* q, magma_int_t ldq, double rho,
    double* dlamda, double* q2, magma_int_t* indx,
    magma_int_t* ctot, double* w, double* s, magma_int_t* indxq,
    magmaDouble_ptr dwork,
    magma_vec_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t* info, magma_queue_t queue );

#endif


/* ////////////////////////////////////////////////////////////////////////////
    -- MAGMA function definitions / Data on GPU
*/

magma_err_t
magma_zgeqrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magma_int_t *info, magma_queue_t *queue );

magma_err_t
magma_zgeqrf2_2q_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t 
magma_zgeqrf2_mgpu( 
    magma_int_t num_gpus, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex_ptr *dlA, magma_int_t ldda, 
    magmaDoubleComplex *tau, 
    magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_zgeqrf_msub( magma_int_t num_subs, magma_int_t num_gpus,
                   magma_int_t m, magma_int_t n,
                   magmaDoubleComplex_ptr *dlA, magma_int_t ldda,
                   magmaDoubleComplex *tau,
                   magma_int_t *info, magma_queue_t *queues);
magma_err_t 
magma_zgetrf2_mgpu(
    magma_int_t num_gpus, 
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset, 
    magmaDoubleComplex_ptr *d_lAT, size_t dlAT_offset, magma_int_t lddat, 
    magma_int_t *ipiv, 
    magmaDoubleComplex_ptr *d_lAP, size_t dlAP_offset, 
    magmaDoubleComplex *a, magma_int_t lda, 
    magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_zgetrf_mgpu(
    magma_int_t num_gpus, 
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex_ptr *d_lA, size_t dlA_offset, magma_int_t ldda, 
    magma_int_t *ipiv, magma_int_t *info, 
    magma_queue_t *queues);

magma_err_t
magma_zgetrf_msub(magma_int_t trans, magma_int_t num_subs, magma_int_t num_gpus,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr *d_lA, size_t dlA_offset, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t *info,
    magma_queue_t *queues);

magma_err_t
magma_zgetrf2_msub(magma_int_t num_subs, magma_int_t num_gpus,
         magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
         magmaDoubleComplex_ptr *d_lAT, size_t dlAT_offset, magma_int_t lddat,
         magma_int_t *ipiv,
         magmaDoubleComplex_ptr *d_panel,
         magmaDoubleComplex_ptr *d_lAP, size_t dlAP_offset,
         magmaDoubleComplex *w, magma_int_t ldw,
         magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_zgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_zgetrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t
magma_zlarfb_gpu(
    int side, int trans, int direct, int storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_ptr dV, size_t dV_offset, magma_int_t ldv,
    magmaDoubleComplex_ptr dT, size_t dT_offset, magma_int_t ldt,
    magmaDoubleComplex_ptr dC, size_t dC_offset, magma_int_t ldc,
    magmaDoubleComplex_ptr dwork, size_t dwork_offset, magma_int_t ldwork,
    magma_queue_t queue );

magma_err_t
magma_zpotrf_gpu(
    int uplo,
    magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_zpotrf2_gpu(
    int uplo,
    magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t 
magma_zpotrf_mgpu(int num_gpus, magma_uplo_t uplo, magma_int_t n, 
        magmaDoubleComplex_ptr *d_lA, size_t dA_offset, 
        magma_int_t ldda, magma_int_t *info, 
        magma_queue_t *queues);

magma_err_t 
magma_zpotrf2_mgpu(int num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
        magma_int_t off_i, magma_int_t off_j, magma_int_t nb, 
        magmaDoubleComplex_ptr *d_lA, size_t dA_offset, magma_int_t ldda, 
        magmaDoubleComplex_ptr *d_lP, magma_int_t lddp, 
        magmaDoubleComplex  *a,    magma_int_t lda, magma_int_t h, 
        magma_int_t *info, magma_queue_t *queues );

magma_err_t
magma_zpotrf_msub(int num_subs, int num_gpus, magma_uplo_t uplo, magma_int_t n,
                  magmaDoubleComplex_ptr *d_lA, size_t dA_offset,
                  magma_int_t ldda, magma_int_t *info,
                  magma_queue_t *queues);

magma_err_t
magma_zpotrf2_msub(int num_subs, int num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                   magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                   magmaDoubleComplex_ptr *d_lA, size_t dA_offset, magma_int_t ldda,
                   magmaDoubleComplex_ptr *d_lP, magma_int_t lddp,
                   magmaDoubleComplex  *a,    magma_int_t lda, magma_int_t h,
                   magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_zpotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_err_t
magma_zposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_err_t
magma_zgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_zgesv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_int_t
magma_zunmqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dC, size_t dC_offset, magma_int_t lddc,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magmaDoubleComplex_ptr dT, size_t dT_offset, magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_zgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT, size_t dT_offset,
    magmaDoubleComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_zgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset,  magma_int_t ldda,
    magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, size_t dT_offset,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, size_t dA_offset,  magma_int_t ldda,
    magmaDoubleComplex_ptr dB, size_t dB_offset,  magma_int_t lddb,
    magmaDoubleComplex *hwork, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zgehrd(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex_ptr dT, size_t dT_offset,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zlabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex_ptr da, size_t da_offset, magma_int_t ldda,
    double *d, double *e, magmaDoubleComplex *tauq, magmaDoubleComplex *taup,
    magmaDoubleComplex *x, magma_int_t ldx,
    magmaDoubleComplex_ptr dx, size_t dx_offset, magma_int_t lddx,
    magmaDoubleComplex *y, magma_int_t ldy,
    magmaDoubleComplex_ptr dy, size_t dy_offset, magma_int_t lddy,
    magma_queue_t queue );

magma_err_t
magma_zlahr2(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaDoubleComplex_ptr da, size_t da_offset,
    magmaDoubleComplex_ptr dv, size_t dv_offset,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *tau, magmaDoubleComplex *t, magma_int_t ldt,
    magmaDoubleComplex *y, magma_int_t ldy,
    magma_queue_t queue );

magma_err_t
magma_zlahru(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex_ptr da, size_t da_offset,
    magmaDoubleComplex_ptr y,  size_t y_offset,
    magmaDoubleComplex_ptr v,  size_t v_offset,
    magmaDoubleComplex_ptr dt, size_t dt_offset,
    magmaDoubleComplex_ptr dwork, size_t dwork_offset,
    magma_queue_t queue );

magma_err_t
magma_zlatrd(
    char uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *a,  magma_int_t lda,
    double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *w,  magma_int_t ldw,
    magmaDoubleComplex_ptr da, size_t da_offset, magma_int_t ldda,
    magmaDoubleComplex_ptr dw, size_t dw_offset, magma_int_t lddw,
    magma_queue_t queue );

magma_err_t
magma_zhetrd(
    char uplo, magma_int_t n,
    magmaDoubleComplex *a, magma_int_t lda,
    double *d, double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_ztrtri_gpu(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info );

magma_int_t
magma_zgetri_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t lda,
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dwork, size_t dwork_offset, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zlauum_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_zpotri_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr a, size_t offset_a, magma_int_t lda,
    magma_int_t *info, magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
    -- MAGMA utility function definitions
*/

void magma_zprint(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda );

void magma_zprint_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue );

#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMA_Z_H */
