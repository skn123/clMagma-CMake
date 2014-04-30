/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zgeqr2x_gpu-v3.cpp normal z -> s, Fri Jan 10 15:51:18 2014

*/
#include "common_magma.h"

extern "C" void
magma_slarfbx_gpu(int m, int k, magmaFloat_ptr V, size_t v_offset, int ldv,
                  magmaFloat_ptr dT, size_t dT_offset, int ldt, magmaFloat_ptr c, size_t c_offset, 
                  magmaFloat_ptr dwork, size_t dwork_offset, magma_queue_t queue);

extern "C" void
magma_slarfgtx_gpu(int n, magmaFloat_ptr dx0, size_t dx0_offset, magmaFloat_ptr dx, size_t dx_offset, 
                   magmaFloat_ptr dtau, size_t dtau_offset, magmaFloat_ptr dxnorm, size_t dxnorm_offset, 
                   magmaFloat_ptr dA, size_t dA_offset, int it,
                   magmaFloat_ptr V, size_t V_offset, int ldv, magmaFloat_ptr T, size_t T_offset, int ldt,
                   magmaFloat_ptr dwork, size_t dwork_offset, magma_queue_t queue);

extern "C" void
magmablas_snrm2(int m, int num, magmaFloat_ptr da, size_t da_offset, magma_int_t ldda, magmaFloat_ptr dxnorm, size_t dxnorm_offset, 
                 magma_queue_t queue);

extern "C" void
magmablas_snrm2_adjust(int k, magmaFloat_ptr xnorm, size_t xnorm_offset, magmaFloat_ptr c, size_t c_offset, magma_queue_t queue);
    
extern "C" void
magmablas_sgemm_reduce(magma_int_t m, magma_int_t n, magma_int_t k,
                       float alpha, const magmaFloat_ptr d_A, size_t d_A_offset, magma_int_t lda,
                       const magmaFloat_ptr d_B, size_t d_B_offset, magma_int_t ldb,
                       float beta,        magmaFloat_ptr d_C, size_t d_C_offset, magma_int_t ldc,
                       magma_queue_t queue);


extern "C" magma_err_t
magma_slarfb2_gpu( magma_int_t m, magma_int_t n, magma_int_t k,
                   const magmaFloat_ptr dV, size_t dV_offset, magma_int_t ldv,
                   const magmaFloat_ptr dT, size_t dT_offset, magma_int_t ldt,
                   magmaFloat_ptr dC, size_t dC_offset, magma_int_t ldc,
                   magmaFloat_ptr dwork, size_t dwork_offset, magma_int_t ldwork, 
                   magma_queue_t queue )
{
    float c_zero    = MAGMA_S_ZERO;
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    if (m <= 0 || n <= 0)
        return MAGMA_SUCCESS;

    // W = C^H V
    // magma_sgemm( MagmaTrans, MagmaNoTrans,
    magmablas_sgemm_reduce(
                           n, k, m,
                           c_one, dC, dC_offset, ldc,
                           dV, dV_offset, ldv,
                           c_zero, dwork, dwork_offset, ldwork, queue);

    // W = W T^H = C^H V T^H
    magma_strmm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                 n, k,
                 c_one, dT, dT_offset, ldt,
                 dwork, dwork_offset, ldwork,
                 queue);

    // C = C - V W^H = C - V T V^H C = (I - V T V^H) C = H C
    magma_sgemm( MagmaNoTrans, MagmaTrans,
                 m, n, k,
                 c_neg_one, dV, dV_offset, ldv,
                 dwork, dwork_offset, ldwork,
                 c_one, dC, dC_offset, ldc, queue );
}

//////////////////////////////////////////////////////////////////////////////

extern "C" magma_err_t
magma_sgeqr2x3_gpu(magma_int_t *m, magma_int_t *n, 
                   magmaFloat_ptr dA, size_t dA_offset, magma_int_t *ldda, 
                   magmaFloat_ptr dtau, size_t dtau_offset, 
                   magmaFloat_ptr dT, size_t dT_offset, 
                   magmaFloat_ptr ddA, size_t ddA_offset, 
                   magmaFloat_ptr dwork, size_t dwork_offset, 
                   magma_int_t *info, magma_queue_t queue)
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose   
    =======   
    SGEQR2 computes a QR factorization of a real m by n matrix A:   
    A = Q * R.

    This expert routine requires two more arguments than the standard 
    sgeqr2, namely, dT and ddA, explained below. The storage for A is 
    also not as in the LAPACK's sgeqr2 routine (see below). 

    The first is used to output the triangular 
    n x n factor T of the block reflector used in the factorization. 
    The second holds the diagonal nxn blocks of A, i.e., the diagonal
    submatrices of R. This routine implements the left looking QR.

    This version adds internal blocking.

    Arguments   
    =========   
    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the m by n matrix A.   
            On exit, the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

            the elements on and above the diagonal of the array   
            contain the min(m,n) by n upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the unitary matrix Q as a   
            product of elementary reflectors (see Further Details).   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    dT      (output) REAL array, dimension N x N.
            Stores the triangular N x N factor T of the block reflector 
            used in the factorization. The lower triangular part is 0.

    ddA     (output) REAL array, dimension N x N.
            Stores the elements of the upper N x N diagonal block of A.
            LAPACK stores this array in A. There are 0s below the diagonal.

    RWORK   (workspace) DOUBLE_PRECISION array, dimension (3 N)

    INFO    (output) INTEGER   
            = 0: successful exit   
            < 0: if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   
    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),   
    and tau in TAU(i).   
    =====================================================================    */

    //#define da_ref(a_1,a_2) ( dA+(a_2)*(*ldda) + (a_1))
    #define da_ref(a_1,a_2) dA, (dA_offset + ((a_2)*(*ldda) + (a_1)))
    #define BLOCK_SIZE 32
    //#define BLOCK_SIZE 16

    static magma_int_t i, k;

    //float *dnorm = dwork;
    magmaFloat_ptr dnorm = dwork;
    size_t dnorm_offset = dwork_offset;
    //float *work = (float *)(dwork+2*(*n));
    magmaFloat_ptr work = (magmaFloat_ptr)dwork;
    size_t work_offset = dwork_offset + 2*(*n);

    *info = 0;
    if (*m < 0) {
        *info = -1;
    } else if (*n < 0) {
        *info = -2;
    } else if (*ldda < max(1,*m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Compute the norms of the trailing columns */
    k = min(*m,*n);
    magmablas_snrm2(*m, k, da_ref(0,0), *ldda, dnorm, dnorm_offset, queue);

    for (int b=0; b < k; b += BLOCK_SIZE) {
        for (i = b; i < min(k, b+BLOCK_SIZE); ++i) {

            /*   Apply H' to A(:,i) from the left                           */    
            if ( i-b > 0){
                magma_queue_sync(queue);
                magma_slarfbx_gpu(*m-b, i-b, da_ref(b, b), *ldda,
                                  dT, (dT_offset+b+b*k), k, da_ref(b, i), work, work_offset, queue);
            }
            /*   Adjust the dnorm[i] to hold the norm of A(i:m,i)           */ 
            if ( i > 0 ){
                magma_queue_sync(queue);
                magmablas_snrm2_adjust(i, dnorm, dnorm_offset+i, da_ref(0, i), queue);
            }
            /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i) 
                1. 1 is not yet put on the diagonal of A
                2. Elements above the diagonal are copied in ddA and
                   the ones in A are set to zero                                         
                3. update T                                                 */
            magma_slarfgtx_gpu(*m-i, da_ref(i, i), da_ref(min(i+1,*m), i), dtau, dtau_offset+i, 
                               dnorm, dnorm_offset+i, ddA, ddA_offset + i + i*(*n), i,
                               da_ref(i,0), *ldda,  dT, dT_offset, k, work, work_offset, queue);
        }
        
        /* Apply the transformations to the trailing matrix. */
        magma_slarfb2_gpu(
                           *m-b, k-i, BLOCK_SIZE,
                           da_ref(b, b), *ldda, dT, dT_offset+b+b*k, k,
                           da_ref(b, i), *ldda, work, work_offset, k-i, queue);
    }
    magma_queue_sync(queue);
    return *info;
} /* magma_sgeqr2 */
