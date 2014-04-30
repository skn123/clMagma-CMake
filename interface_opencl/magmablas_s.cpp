/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 * @author Chongxiao Cao
 * @generated from magmablas_z.cpp normal z -> s, Fri Jan 10 15:51:19 2014
 */

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"

#define PRECISION_s
#ifdef HAVE_clAmdBlas

// AMD is inconsistent in their function names: it's Ssymv but DsymvEx.
// Use SsymvEx name below, since DsymvEx requires the Ex, but rename to Ssymv.
#if defined(PRECISION_z) || defined(PRECISION_c)
#define clAmdBlasSsymvEx  clAmdBlasSsymv
#define clAmdBlasSsyrkEx  clAmdBlasSsyrk
#define clAmdBlasSsyr2kEx  clAmdBlasSsyr2k
#endif

// ========================================
// globals, defined in interface.c
extern cl_platform_id gPlatform;
extern cl_context     gContext;

extern magma_event_t  *gevent;

// ========================================
// copying sub-matrices (contiguous columns)
// OpenCL takes queue even for blocking transfers, oddly.
magma_err_t
magma_ssetmatrix(
    magma_int_t m, magma_int_t n,
    float const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaFloat_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    size_t buffer_origin[3] = { dA_offset*sizeof(float), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(float), 0, 0 };
    size_t region[3]        = { m*sizeof(float), n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
        queue, dA_dst, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(float), 0,
        ldha*sizeof(float), 0,
        hA_src, 0, NULL, gevent );
    return err;
}

// --------------------
magma_err_t
magma_ssetvector(
    magma_int_t n,
    float const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaFloat_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if (n <= 0)
       return MAGMA_SUCCESS;

    cl_int err;
    if(incx == 1 && incy == 1) {
        err = clEnqueueWriteBuffer(
            queue, dA_dst, CL_TRUE,
            dA_offset*sizeof(float), n*sizeof(float),
            hA_src+hA_offset, 0, NULL, gevent);
        return err;
    } else {
        magma_int_t ldha = incx;
        magma_int_t ldda = incy;
        cl_int err = magma_ssetmatrix(1, n,
            hA_src, hA_offset, ldha,
            dA_dst, dA_offset, ldda,
            queue);
        return err;
    }
}

// --------------------
magma_err_t
magma_sgetmatrix(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    float*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
      return MAGMA_SUCCESS;

    size_t buffer_origin[3] = { dA_offset*sizeof(float), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(float), 0, 0 };
    size_t region[3]        = { m*sizeof(float), n, 1 };
    cl_int err = clEnqueueReadBufferRect(
        queue, dA_src, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(float), 0,
        ldha*sizeof(float), 0,
        hA_dst, 0, NULL, gevent );
    return err;
}

// --------------------
magma_err_t
magma_sgetvector(
    magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    float*             hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( n <= 0 )
       return MAGMA_SUCCESS;

    cl_int err;
    if(incx ==1 && incy ==1) {
        err = clEnqueueReadBuffer(
            queue, dA_src, CL_TRUE,
            dA_offset*sizeof(float), n*sizeof(float),
            hA_dst+hA_offset, 0, NULL, gevent);
        return err;
    } else {
        magma_int_t ldda = incx;
        magma_int_t ldha = incy;
        err = magma_sgetmatrix(1, n,
            dA_src, dA_offset, ldda,
            hA_dst, hA_offset, ldha,
            queue);
        return err;
    }
}

// --------------------
magma_err_t
magma_sgetvector_async(
    magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    float*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event )
{
    if ( n <= 0 )
       return MAGMA_SUCCESS;

    cl_int err;
    if(incx ==1 && incy ==1) {
        err = clEnqueueReadBuffer(
            queue, dA_src, CL_FALSE,
            dA_offset*sizeof(float), n*sizeof(float),
            hA_dst+hA_offset, 0, NULL, event);
        return err;
    } else {
        magma_int_t ldda = incx;
        magma_int_t ldha = incy;
        err = magma_sgetmatrix_async(1, n,
            dA_src, dA_offset, ldda,
            hA_dst, hA_offset, ldha,
            queue, event);
        return err;
    }
}

// --------------------
magma_err_t
magma_ssetvector_async(
    magma_int_t n,
    float const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaFloat_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event )
{
    if ( n <= 0 )
       return MAGMA_SUCCESS;

    cl_int err;
    if(incx == 1 && incy == 1) {
        err = clEnqueueWriteBuffer(
            queue, dA_dst, CL_FALSE,
            dA_offset*sizeof(float), n*sizeof(float),
            hA_src+hA_offset, 0, NULL, event);
        return err;
    } else {
        magma_int_t ldha = incx;
        magma_int_t ldda = incy;
        cl_int err = magma_ssetmatrix_async(1, n,
            hA_src, hA_offset, ldha,
            dA_dst, dA_offset, ldda,
            queue, event);
        return err;
    }
}

// --------------------
magma_err_t
magma_ssetmatrix_async(
    magma_int_t m, magma_int_t n,
    float const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaFloat_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue, magma_event_t *event )
{
    if ( m<=0 || n <= 0 )
       return MAGMA_SUCCESS;

    size_t buffer_origin[3] = { dA_offset*sizeof(float), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(float), 0, 0 };
    size_t region[3]        = { m*sizeof(float), n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
        queue, dA_dst, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(float), 0,
        ldha*sizeof(float), 0,
        hA_src, 0, NULL, event );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_sgetmatrix_async(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    float*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue, magma_event_t *event )
{
    if (m<=0 || n <= 0)
      return MAGMA_SUCCESS;

    size_t buffer_origin[3] = { dA_offset*sizeof(float), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(float), 0, 0 };
    size_t region[3]        = { m*sizeof(float), n, 1 };
    cl_int err = clEnqueueReadBufferRect(
        queue, dA_src, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(float), 0,
        ldha*sizeof(float), 0,
        hA_dst, 0, NULL, event );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_scopymatrix(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr    dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
{
    if ( m<=0 || n <= 0 )
       return MAGMA_SUCCESS;

    size_t src_origin[3] = { dA_offset*sizeof(float), 0, 0 };
    size_t dst_orig[3]   = { dB_offset*sizeof(float), 0, 0 };
    size_t region[3]        = { m*sizeof(float), n, 1 };
    cl_int err = clEnqueueCopyBufferRect(
        queue, dA_src, dB_dst,
        src_origin, dst_orig, region,
        ldda*sizeof(float), 0,
        lddb*sizeof(float), 0,
        0, NULL, gevent );
    return err;
}
// ========================================
// BLAS functions
magma_err_t
magma_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    float beta,  magmaFloat_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0 || k<=0)
      return MAGMA_SUCCESS;

    cl_int err = clAmdBlasSgemmEx(
        clAmdBlasColumnMajor,
        amdblas_trans_const( transA ),
        amdblas_trans_const( transB ),
        m, n, k,
        alpha, dA, dA_offset, lda,
               dB, dB_offset, ldb,
        beta,  dC, dC_offset, ldc,
        1, &queue, 0, NULL, gevent );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_sgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    float beta,  magmaFloat_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasSgemvEx(
        clAmdBlasColumnMajor,
        amdblas_trans_const( transA ),
        m, n,
        alpha, dA, dA_offset, lda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        1, &queue, 0, NULL, gevent );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    float beta,  magmaFloat_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasSsymm(
        clAmdBlasColumnMajor,
        amdblas_side_const( side ),
        amdblas_uplo_const( uplo ),
        m, n,
        alpha, dA, dA_offset, lda,
               dB, dB_offset, ldb,
        beta,  dC, dC_offset, ldc,
        1, &queue, 0, NULL, gevent );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_ssymv(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
    float beta,  magmaFloat_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( n <= 0 )
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasSsymvEx(
        clAmdBlasColumnMajor,
        amdblas_uplo_const( uplo ),
        n,
        alpha, dA, dA_offset, lda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        1, &queue, 0, NULL, gevent );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
    float beta,  magmaFloat_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue )
{
    if (n<=0 || k <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasSsyrkEx(
        clAmdBlasColumnMajor,
        amdblas_uplo_const( uplo ),
        amdblas_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, lda,
        beta,  dC, dC_offset, ldc,
        1, &queue, 0, NULL, gevent );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasStrsmEx(
        clAmdBlasColumnMajor,
        amdblas_side_const( side ),
        amdblas_uplo_const( uplo ),
        amdblas_trans_const( trans ),
        amdblas_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, lda,
               dB, dB_offset, ldb,
        1, &queue, 0, NULL, gevent );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_strsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaFloat_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return MAGMA_SUCCESS;

    cl_int err = clAmdBlasStrsv(
        clAmdBlasColumnMajor,
        amdblas_uplo_const( uplo ),
        amdblas_trans_const( trans ),
        amdblas_diag_const( diag ),
        n,
        dA, dA_offset, lda,
        dx, dx_offset, incx,
        1, &queue, 0, NULL, gevent );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasStrmmEx(
        clAmdBlasColumnMajor,
        amdblas_side_const( side ),
        amdblas_uplo_const( uplo ),
        amdblas_trans_const( trans ),
        amdblas_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, lda,
               dB, dB_offset, ldb,
        1, &queue, 0, NULL, gevent );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    float beta, magmaFloat_ptr dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue)
{
     if (n<=0 || k <= 0)
        return MAGMA_SUCCESS;

     cl_int err = clAmdBlasSsyr2kEx(
         clAmdBlasColumnMajor,
         amdblas_uplo_const( uplo ),
         amdblas_trans_const( trans ),
         n, k,
         alpha, dA, dA_offset, lda,
         dB, dB_offset, ldb,
         beta, dC, dC_offset, ldc,
         1, &queue, 0, NULL, gevent );
     clFlush(queue);
     return err;
}

#endif // HAVE_clAmdBlas
