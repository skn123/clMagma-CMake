/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 * @generated from magma_z.h normal z -> s, Fri Jan 10 15:51:16 2014
 */

#ifndef MAGMA_S_H
#define MAGMA_S_H

#include "magma_types.h"

#define PRECISION_s

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
magma_int_t magma_get_spotrf_nb( magma_int_t m );
magma_int_t magma_get_sgetrf_nb( magma_int_t m );
magma_int_t magma_get_sgetri_nb( magma_int_t m );
magma_int_t magma_get_sgeqrf_nb( magma_int_t m );
magma_int_t magma_get_sgeqlf_nb( magma_int_t m );
magma_int_t magma_get_sgehrd_nb( magma_int_t m );
magma_int_t magma_get_ssytrd_nb( magma_int_t m );
magma_int_t magma_get_sgelqf_nb( magma_int_t m );
magma_int_t magma_get_sgebrd_nb( magma_int_t m );
magma_int_t magma_get_ssygst_nb( magma_int_t m );
magma_int_t magma_get_sgesvd_nb( magma_int_t m );

/* ////////////////////////////////////////////////////////////////////////////
    -- MAGMA function definitions / Data on CPU
*/

magma_err_t
magma_sgebrd(
    magma_int_t m, magma_int_t n,
    float *a, magma_int_t lda, float *d, float *e,
    float *tauq, float *taup,
    float *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_err_t 
magma_sgeqrf( 
    magma_int_t m, magma_int_t n, float *A,
    magma_int_t lda, float *tau, float *work,
    magma_int_t lwork, magma_int_t *info, magma_queue_t *queue);

magma_err_t 
magma_sgesv ( 
    magma_int_t n, magma_int_t nrhs,
    float *A, magma_int_t lda, magma_int_t *ipiv,
    float *B, magma_int_t ldb, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t 
magma_sgetrf( 
    magma_int_t m, magma_int_t n, float *A,
    magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t *queue );

magma_err_t
magma_sposv (
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t
magma_spotrf(
    magma_uplo_t uplo, magma_int_t n, float *A,
    magma_int_t lda, magma_int_t *info,
    magma_queue_t *queue );

magma_err_t
magma_sorghr(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *a, magma_int_t lda,
    float *tau,
    magmaFloat_ptr dT, size_t dT_offset,
    magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_sorgqr(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *a, magma_int_t lda,
    float *tau,
    magmaFloat_ptr dT, size_t dT_offset,
    magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sormtr(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    float *a,    magma_int_t lda,
    float *tau,
    float *c,    magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sormqr(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *a,    magma_int_t lda,
    float *tau,
    float *c,    magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sormql(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *a, magma_int_t lda,
    float *tau,
    float *c, magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

#if defined(PRECISION_z) || defined(PRECISION_c)

magma_int_t
magma_sgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    float *a, magma_int_t lda,
    float *geev_w_array,
    float *vl, magma_int_t ldvl,
    float *vr, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sgesvd(
    char jobu, char jobvt, magma_int_t m, magma_int_t n,
    float *a,    magma_int_t lda, float *s,
    float *u,    magma_int_t ldu,
    float *vt,   magma_int_t ldvt,
    float *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_ssyevd(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    float *a, magma_int_t lda,
    float *w,
    float *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sstedx(
    magma_vec_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float* d, float* e,
    float* z, magma_int_t ldz,
    float* rwork, magma_int_t lrwork,
    magma_int_t* iwork, magma_int_t liwork,
    magmaFloat_ptr dwork,
    magma_int_t* info, magma_queue_t queue );

#else

magma_int_t
magma_sgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    float *a, magma_int_t lda,
    float *WR, float *WI,
    float *vl, magma_int_t ldvl,
    float *vr, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sgesvd(
    char jobu, char jobvt, magma_int_t m, magma_int_t n,
    float *a,    magma_int_t lda, float *s,
    float *u,    magma_int_t ldu,
    float *vt,   magma_int_t ldvt,
    float *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_ssyevd(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    float *a, magma_int_t lda,
    float *w,
    float *work, magma_int_t lwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sstedx(
    magma_vec_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float* d, float* e,
    float* z, magma_int_t ldz,
    float* rwork, magma_int_t lrwork,
    magma_int_t* iwork, magma_int_t liwork,
    magmaFloat_ptr dwork,
    magma_int_t* info, magma_queue_t queue );

magma_int_t
magma_slaex0(
    magma_int_t n, float* d, float* e, float* q, magma_int_t ldq,
    float* work, magma_int_t* iwork, magmaFloat_ptr dwork,
    magma_vec_t range, float vl, float vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t* info, magma_queue_t queue );

magma_int_t
magma_slaex1(
    magma_int_t n, float* d, float* q, magma_int_t ldq,
    magma_int_t* indxq, float rho, magma_int_t cutpnt,
    float* work, magma_int_t* iwork, magmaFloat_ptr dwork,
    magma_vec_t range, float vl, float vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t* info, magma_queue_t queue );


magma_int_t
magma_slaex3(
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
magma_sgeqrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    float *tau,
    magma_int_t *info, magma_queue_t *queue );

magma_err_t
magma_sgeqrf2_2q_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    float *tau,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t 
magma_sgeqrf2_mgpu( 
    magma_int_t num_gpus, magma_int_t m, magma_int_t n, 
    magmaFloat_ptr *dlA, magma_int_t ldda, 
    float *tau, 
    magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_sgeqrf_msub( magma_int_t num_subs, magma_int_t num_gpus,
                   magma_int_t m, magma_int_t n,
                   magmaFloat_ptr *dlA, magma_int_t ldda,
                   float *tau,
                   magma_int_t *info, magma_queue_t *queues);
magma_err_t 
magma_sgetrf2_mgpu(
    magma_int_t num_gpus, 
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset, 
    magmaFloat_ptr *d_lAT, size_t dlAT_offset, magma_int_t lddat, 
    magma_int_t *ipiv, 
    magmaFloat_ptr *d_lAP, size_t dlAP_offset, 
    float *a, magma_int_t lda, 
    magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_sgetrf_mgpu(
    magma_int_t num_gpus, 
    magma_int_t m, magma_int_t n, 
    magmaFloat_ptr *d_lA, size_t dlA_offset, magma_int_t ldda, 
    magma_int_t *ipiv, magma_int_t *info, 
    magma_queue_t *queues);

magma_err_t
magma_sgetrf_msub(magma_int_t trans, magma_int_t num_subs, magma_int_t num_gpus,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr *d_lA, size_t dlA_offset, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t *info,
    magma_queue_t *queues);

magma_err_t
magma_sgetrf2_msub(magma_int_t num_subs, magma_int_t num_gpus,
         magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
         magmaFloat_ptr *d_lAT, size_t dlAT_offset, magma_int_t lddat,
         magma_int_t *ipiv,
         magmaFloat_ptr *d_panel,
         magmaFloat_ptr *d_lAP, size_t dlAP_offset,
         float *w, magma_int_t ldw,
         magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_sgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_sgetrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t
magma_slarfb_gpu(
    int side, int trans, int direct, int storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dV, size_t dV_offset, magma_int_t ldv,
    magmaFloat_ptr dT, size_t dT_offset, magma_int_t ldt,
    magmaFloat_ptr dC, size_t dC_offset, magma_int_t ldc,
    magmaFloat_ptr dwork, size_t dwork_offset, magma_int_t ldwork,
    magma_queue_t queue );

magma_err_t
magma_spotrf_gpu(
    int uplo,
    magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_spotrf2_gpu(
    int uplo,
    magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t* queues );

magma_err_t 
magma_spotrf_mgpu(int num_gpus, magma_uplo_t uplo, magma_int_t n, 
        magmaFloat_ptr *d_lA, size_t dA_offset, 
        magma_int_t ldda, magma_int_t *info, 
        magma_queue_t *queues);

magma_err_t 
magma_spotrf2_mgpu(int num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
        magma_int_t off_i, magma_int_t off_j, magma_int_t nb, 
        magmaFloat_ptr *d_lA, size_t dA_offset, magma_int_t ldda, 
        magmaFloat_ptr *d_lP, magma_int_t lddp, 
        float  *a,    magma_int_t lda, magma_int_t h, 
        magma_int_t *info, magma_queue_t *queues );

magma_err_t
magma_spotrf_msub(int num_subs, int num_gpus, magma_uplo_t uplo, magma_int_t n,
                  magmaFloat_ptr *d_lA, size_t dA_offset,
                  magma_int_t ldda, magma_int_t *info,
                  magma_queue_t *queues);

magma_err_t
magma_spotrf2_msub(int num_subs, int num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                   magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                   magmaFloat_ptr *d_lA, size_t dA_offset, magma_int_t ldda,
                   magmaFloat_ptr *d_lP, magma_int_t lddp,
                   float  *a,    magma_int_t lda, magma_int_t h,
                   magma_int_t *info, magma_queue_t *queues);

magma_err_t
magma_spotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_err_t
magma_sposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_err_t
magma_sgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaFloat_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_sgesv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaFloat_ptr dB, size_t dB_offset, magma_int_t lddb,
    magma_err_t *info, magma_queue_t queue );

magma_int_t
magma_sormqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dC, size_t dC_offset, magma_int_t lddc,
    float *hwork, magma_int_t lwork,
    magmaFloat_ptr dT, size_t dT_offset, magma_int_t nb,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_sgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dT, size_t dT_offset,
    magmaFloat_ptr dB, size_t dB_offset, magma_int_t lddb,
    float *hwork, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_sgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset,  magma_int_t ldda,
    float *tau, magmaFloat_ptr dT, size_t dT_offset,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, size_t dA_offset,  magma_int_t ldda,
    magmaFloat_ptr dB, size_t dB_offset,  magma_int_t lddb,
    float *hwork, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_sgehrd(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *a, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magmaFloat_ptr dT, size_t dT_offset,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_slabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    float *a, magma_int_t lda,
    magmaFloat_ptr da, size_t da_offset, magma_int_t ldda,
    float *d, float *e, float *tauq, float *taup,
    float *x, magma_int_t ldx,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t lddx,
    float *y, magma_int_t ldy,
    magmaFloat_ptr dy, size_t dy_offset, magma_int_t lddy,
    magma_queue_t queue );

magma_err_t
magma_slahr2(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaFloat_ptr da, size_t da_offset,
    magmaFloat_ptr dv, size_t dv_offset,
    float *a, magma_int_t lda,
    float *tau, float *t, magma_int_t ldt,
    float *y, magma_int_t ldy,
    magma_queue_t queue );

magma_err_t
magma_slahru(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    float *a, magma_int_t lda,
    magmaFloat_ptr da, size_t da_offset,
    magmaFloat_ptr y,  size_t y_offset,
    magmaFloat_ptr v,  size_t v_offset,
    magmaFloat_ptr dt, size_t dt_offset,
    magmaFloat_ptr dwork, size_t dwork_offset,
    magma_queue_t queue );

magma_err_t
magma_slatrd(
    char uplo, magma_int_t n, magma_int_t nb,
    float *a,  magma_int_t lda,
    float *e, float *tau,
    float *w,  magma_int_t ldw,
    magmaFloat_ptr da, size_t da_offset, magma_int_t ldda,
    magmaFloat_ptr dw, size_t dw_offset, magma_int_t lddw,
    magma_queue_t queue );

magma_err_t
magma_ssytrd(
    char uplo, magma_int_t n,
    float *a, magma_int_t lda,
    float *d, float *e, float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_strtri_gpu(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info );

magma_int_t
magma_sgetri_gpu(
    magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t lda,
    magma_int_t *ipiv,
    magmaFloat_ptr dwork, size_t dwork_offset, magma_int_t lwork,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_slauum_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_spotri_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr a, size_t offset_a, magma_int_t lda,
    magma_int_t *info, magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
    -- MAGMA utility function definitions
*/

void magma_sprint(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda );

void magma_sprint_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue );

#ifdef __cplusplus
}
#endif

#undef PRECISION_s
#endif /* MAGMA_S_H */
