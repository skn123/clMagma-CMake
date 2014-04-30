/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 * @generated from magma_z.h normal z -> c, Fri Jan 10 15:51:16 2014
 */

#ifndef MAGMA_C_H
#define MAGMA_C_H

#include "magma_types.h"

#define PRECISION_c

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
magma_int_t magma_get_cpotrf_nb( magma_int_t m );
magma_int_t magma_get_cgetrf_nb( magma_int_t m );
magma_int_t magma_get_cgetri_nb( magma_int_t m );
magma_int_t magma_get_cgeqrf_nb( magma_int_t m );
magma_int_t magma_get_cgeqlf_nb( magma_int_t m );
magma_int_t magma_get_cgehrd_nb( magma_int_t m );
magma_int_t magma_get_chetrd_nb( magma_int_t m );
magma_int_t magma_get_cgelqf_nb( magma_int_t m );
magma_int_t magma_get_cgebrd_nb( magma_int_t m );
magma_int_t magma_get_chegst_nb( magma_int_t m );
magma_int_t magma_get_cgesvd_nb( magma_int_t m );

/* ////////////////////////////////////////////////////////////////////////////
    -- MAGMA function definitions / Data on CPU
*/

magma_err_t
magma_cgebrd(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *a, magma_int_t lda, float *d, float *e,
    magmaFloatComplex *tauq, magmaFloatComplex *taup,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_err_t 
magma_cgeqrf( 
    magma_int_t m, magma_int_t n, magmaFloatComplex *A,
    magma_int_t lda, magmaFloatComplex *tau, magmaFloatComplex *work,
    magma_int_t lwork, magma_int_t *info, magma_queue_t *queue);

magma_err_t 
magma_cgesv ( 
    magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magmaFloatComplex *B, magma_int_t ldb, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t 
magma_cgetrf( 
    magma_int_t m, magma_int_t n, magmaFloatComplex *A,
    magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t *queue );

magma_err_t
magma_cposv (
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t
magma_cpotrf(
    magma_uplo_t uplo, magma_int_t n, magmaFloatComplex *A,
    magma_int_t lda, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t
magma_cunghr(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT, size_t dT_offset,
    magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_cungqr(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT, size_t dT_offset,
    magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cunmtr(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *a,    magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *c,    magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cunmqr(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *a,    magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *c,    magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cunmql(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *c, magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

#if defined(PRECISION_z) || defined(PRECISION_c)

magma_int_t
magma_cgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex *geev_w_array,
    magmaFloatComplex *vl, magma_int_t ldvl,
    magmaFloatComplex *vr, magma_int_t ldvr,
    magmaFloatComplex *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cgesvd(
    char jobu, char jobvt, magma_int_t m, magma_int_t n,
    magmaFloatComplex *a,    magma_int_t lda, float *s,
    magmaFloatComplex *u,    magma_int_t ldu,
    magmaFloatComplex *vt,   magma_int_t ldvt,
    magmaFloatComplex *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cheevd(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *a, magma_int_t lda,
    float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cstedx(
    magma_vec_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float* d, float* e,
    magmaFloatComplex* z, magma_int_t ldz,
    float* rwork, magma_int_t lrwork,
    magma_int_t* iwork, magma_int_t liwork,
    magmaFloat_ptr dwork,
    magma_int_t* info, magma_queue_t queue );

#else

magma_int_t
magma_cgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex *WR, float *WI,
    magmaFloatComplex *vl, magma_int_t ldvl,
    magmaFloatComplex *vr, magma_int_t ldvr,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cgesvd(
    char jobu, char jobvt, magma_int_t m, magma_int_t n,
    magmaFloatComplex *a,    magma_int_t lda, float *s,
    magmaFloatComplex *u,    magma_int_t ldu,
    magmaFloatComplex *vt,   magma_int_t ldvt,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cheevd(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *a, magma_int_t lda,
    float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cstedx(
    magma_vec_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float* d, float* e,
    float* z, magma_int_t ldz,
    float* rwork, magma_int_t lrwork,
    magma_int_t* iwork, magma_int_t liwork,
    magmaFloat_ptr dwork,
    magma_int_t* info, magma_queue_t queue );

magma_int_t
magma_claex0(
    magma_int_t n, float* d, float* e, float* q, magma_int_t ldq,
    float* work, magma_int_t* iwork, magmaFloat_ptr dwork,
    magma_vec_t range, float vl, float vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t* info, magma_queue_t queue );

magma_int_t
magma_claex1(
    magma_int_t n, float* d, float* q, magma_int_t ldq,
    magma_int_t* indxq, float rho, magma_int_t cutpnt,
    float* work, magma_int_t* iwork, magmaFloat_ptr dwork,
    magma_vec_t range, float vl, float vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t* info, magma_queue_t queue );


magma_int_t
magma_claex3(
    magma_int_t k, magma_int_t n, magma_int_t n1, float* d,
    float* q, magma_int_t ldq, float rho,
    float* dlamda, float* q2, magma_int_t* indx,
    magma_int_t* ctot, float* w, float* s, magma_int_t* indxq,
    magmaFloat_ptr dwork,
    magma_vec_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t* info, magma_queue_t queue );

#endif


/* ////////////////////////////////////////////////////////////////////////////
    -- MAGMA function definitions / Data on GPU
*/

magma_err_t
magma_cgeqrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex *tau,
    magma_int_t *info, magma_queue_t *queue );

magma_err_t
magma_cgeqrf2_2q_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex *tau,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t 
magma_cgeqrf2_mgpu( 
    magma_int_t num_gpus, magma_int_t m, magma_int_t n, 
    magmaFloatComplex_ptr *dlA, magma_int_t ldda, 
    magmaFloatComplex *tau, 
    magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_cgeqrf_msub( magma_int_t num_subs, magma_int_t num_gpus,
                   magma_int_t m, magma_int_t n,
                   magmaFloatComplex_ptr *dlA, magma_int_t ldda,
                   magmaFloatComplex *tau,
                   magma_int_t *info, magma_queue_t *queues);
magma_err_t 
magma_cgetrf2_mgpu(
    magma_int_t num_gpus, 
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset, 
    magmaFloatComplex_ptr *d_lAT, size_t dlAT_offset, magma_int_t lddat, 
    magma_int_t *ipiv, 
    magmaFloatComplex_ptr *d_lAP, size_t dlAP_offset, 
    magmaFloatComplex *a, magma_int_t lda, 
    magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_cgetrf_mgpu(
    magma_int_t num_gpus, 
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex_ptr *d_lA, size_t dlA_offset, magma_int_t ldda, 
    magma_int_t *ipiv, magma_int_t *info, 
    magma_queue_t *queues);

magma_err_t
magma_cgetrf_msub(magma_int_t trans, magma_int_t num_subs, magma_int_t num_gpus,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr *d_lA, size_t dlA_offset, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t *info,
    magma_queue_t *queues);

magma_err_t
magma_cgetrf2_msub(magma_int_t num_subs, magma_int_t num_gpus,
         magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
         magmaFloatComplex_ptr *d_lAT, size_t dlAT_offset, magma_int_t lddat,
         magma_int_t *ipiv,
         magmaFloatComplex_ptr *d_panel,
         magmaFloatComplex_ptr *d_lAP, size_t dlAP_offset,
         magmaFloatComplex *w, magma_int_t ldw,
         magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_cgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_cgetrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t
magma_clarfb_gpu(
    int side, int trans, int direct, int storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dV, size_t dV_offset, magma_int_t ldv,
    magmaFloatComplex_ptr dT, size_t dT_offset, magma_int_t ldt,
    magmaFloatComplex_ptr dC, size_t dC_offset, magma_int_t ldc,
    magmaFloatComplex_ptr dwork, size_t dwork_offset, magma_int_t ldwork,
    magma_queue_t queue );

magma_err_t
magma_cpotrf_gpu(
    int uplo,
    magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_cpotrf2_gpu(
    int uplo,
    magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t 
magma_cpotrf_mgpu(int num_gpus, magma_uplo_t uplo, magma_int_t n, 
        magmaFloatComplex_ptr *d_lA, size_t dA_offset, 
        magma_int_t ldda, magma_int_t *info, 
        magma_queue_t *queues);

magma_err_t 
magma_cpotrf2_mgpu(int num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
        magma_int_t off_i, magma_int_t off_j, magma_int_t nb, 
        magmaFloatComplex_ptr *d_lA, size_t dA_offset, magma_int_t ldda, 
        magmaFloatComplex_ptr *d_lP, magma_int_t lddp, 
        magmaFloatComplex  *a,    magma_int_t lda, magma_int_t h, 
        magma_int_t *info, magma_queue_t *queues );

magma_err_t
magma_cpotrf_msub(int num_subs, int num_gpus, magma_uplo_t uplo, magma_int_t n,
                  magmaFloatComplex_ptr *d_lA, size_t dA_offset,
                  magma_int_t ldda, magma_int_t *info,
                  magma_queue_t *queues);

magma_err_t
magma_cpotrf2_msub(int num_subs, int num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                   magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                   magmaFloatComplex_ptr *d_lA, size_t dA_offset, magma_int_t ldda,
                   magmaFloatComplex_ptr *d_lP, magma_int_t lddp,
                   magmaFloatComplex  *a,    magma_int_t lda, magma_int_t h,
                   magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_cpotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_err_t
magma_cposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_err_t
magma_cgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaFloatComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_cgesv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaFloatComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_int_t
magma_cunmqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dC, size_t dC_offset, magma_int_t lddc,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magmaFloatComplex_ptr dT, size_t dT_offset, magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_cgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT, size_t dT_offset,
    magmaFloatComplex_ptr dB, size_t dB_offset, magma_int_t lddb,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_cgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset,  magma_int_t ldda,
    magmaFloatComplex *tau, magmaFloatComplex_ptr dT, size_t dT_offset,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, size_t dA_offset,  magma_int_t ldda,
    magmaFloatComplex_ptr dB, size_t dB_offset,  magma_int_t lddb,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cgehrd(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magmaFloatComplex_ptr dT, size_t dT_offset,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_clabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex_ptr da, size_t da_offset, magma_int_t ldda,
    float *d, float *e, magmaFloatComplex *tauq, magmaFloatComplex *taup,
    magmaFloatComplex *x, magma_int_t ldx,
    magmaFloatComplex_ptr dx, size_t dx_offset, magma_int_t lddx,
    magmaFloatComplex *y, magma_int_t ldy,
    magmaFloatComplex_ptr dy, size_t dy_offset, magma_int_t lddy,
    magma_queue_t queue );

magma_err_t
magma_clahr2(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaFloatComplex_ptr da, size_t da_offset,
    magmaFloatComplex_ptr dv, size_t dv_offset,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex *tau, magmaFloatComplex *t, magma_int_t ldt,
    magmaFloatComplex *y, magma_int_t ldy,
    magma_queue_t queue );

magma_err_t
magma_clahru(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaFloatComplex *a, magma_int_t lda,
    magmaFloatComplex_ptr da, size_t da_offset,
    magmaFloatComplex_ptr y,  size_t y_offset,
    magmaFloatComplex_ptr v,  size_t v_offset,
    magmaFloatComplex_ptr dt, size_t dt_offset,
    magmaFloatComplex_ptr dwork, size_t dwork_offset,
    magma_queue_t queue );

magma_err_t
magma_clatrd(
    char uplo, magma_int_t n, magma_int_t nb,
    magmaFloatComplex *a,  magma_int_t lda,
    float *e, magmaFloatComplex *tau,
    magmaFloatComplex *w,  magma_int_t ldw,
    magmaFloatComplex_ptr da, size_t da_offset, magma_int_t ldda,
    magmaFloatComplex_ptr dw, size_t dw_offset, magma_int_t lddw,
    magma_queue_t queue );

magma_err_t
magma_chetrd(
    char uplo, magma_int_t n,
    magmaFloatComplex *a, magma_int_t lda,
    float *d, float *e, magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_ctrtri_gpu(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info );

magma_int_t
magma_cgetri_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t lda,
    magma_int_t *ipiv,
    magmaFloatComplex_ptr dwork, size_t dwork_offset, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_clauum_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_cpotri_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr a, size_t offset_a, magma_int_t lda,
    magma_int_t *info, magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
    -- MAGMA utility function definitions
*/

void magma_cprint(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda );

void magma_cprint_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue );

#ifdef __cplusplus
}
#endif

#undef PRECISION_c
#endif /* MAGMA_C_H */
