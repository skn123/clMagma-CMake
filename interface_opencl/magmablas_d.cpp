/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 * @author Chongxiao Cao
 * @generated from magmablas_z.cpp normal z -> d, Fri Jan 10 15:51:19 2014
 */

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"

#define PRECISION_d
#ifdef HAVE_clAmdBlas

// AMD is inconsistent in their function names: it's Dsymv but DsymvEx.
// Use DsymvEx name below, since DsymvEx requires the Ex, but rename to Dsymv.
#if defined(PRECISION_z) || defined(PRECISION_c)
#define clAmdBlasDsymvEx  clAmdBlasDsymv
#define clAmdBlasDsyrkEx  clAmdBlasDsyrk
#define clAmdBlasDsyr2kEx  clAmdBlasDsyr2k
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
magma_dsetmatrix(
    magma_int_t m, magma_int_t n,
    double const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaDouble_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    size_t buffer_origin[3] = { dA_offset*sizeof(double), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(double), 0, 0 };
    size_t region[3]        = { m*sizeof(double), n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
        queue, dA_dst, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(double), 0,
        ldha*sizeof(double), 0,
        hA_src, 0, NULL, gevent );
    return err;
}

// --------------------
magma_err_t
magma_dsetvector(
    magma_int_t n,
    double const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaDouble_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if (n <= 0)
       return MAGMA_SUCCESS;

    cl_int err;
    if(incx == 1 && incy == 1) {
        err = clEnqueueWriteBuffer(
            queue, dA_dst, CL_TRUE,
            dA_offset*sizeof(double), n*sizeof(double),
            hA_src+hA_offset, 0, NULL, gevent);
        return err;
    } else {
        magma_int_t ldha = incx;
        magma_int_t ldda = incy;
        cl_int err = magma_dsetmatrix(1, n,
            hA_src, hA_offset, ldha,
            dA_dst, dA_offset, ldda,
            queue);
        return err;
    }
}

// --------------------
magma_err_t
magma_dgetmatrix(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    double*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
      return MAGMA_SUCCESS;

    size_t buffer_origin[3] = { dA_offset*sizeof(double), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(double), 0, 0 };
    size_t region[3]        = { m*sizeof(double), n, 1 };
    cl_int err = clEnqueueReadBufferRect(
        queue, dA_src, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(double), 0,
        ldha*sizeof(double), 0,
        hA_dst, 0, NULL, gevent );
    return err;
}

// --------------------
magma_err_t
magma_dgetvector(
    magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    double*             hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( n <= 0 )
       return MAGMA_SUCCESS;

    cl_int err;
    if(incx ==1 && incy ==1) {
        err = clEnqueueReadBuffer(
            queue, dA_src, CL_TRUE,
            dA_offset*sizeof(double), n*sizeof(double),
            hA_dst+hA_offset, 0, NULL, gevent);
        return err;
    } else {
        magma_int_t ldda = incx;
        magma_int_t ldha = incy;
        err = magma_dgetmatrix(1, n,
            dA_src, dA_offset, ldda,
            hA_dst, hA_offset, ldha,
            queue);
        return err;
    }
}

// --------------------
magma_err_t
magma_dgetvector_async(
    magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t incx,
    double*          hA_dst, size_t hA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event )
{
    if ( n <= 0 )
       return MAGMA_SUCCESS;

    cl_int err;
    if(incx ==1 && incy ==1) {
        err = clEnqueueReadBuffer(
            queue, dA_src, CL_FALSE,
            dA_offset*sizeof(double), n*sizeof(double),
            hA_dst+hA_offset, 0, NULL, event);
        return err;
    } else {
        magma_int_t ldda = incx;
        magma_int_t ldha = incy;
        err = magma_dgetmatrix_async(1, n,
            dA_src, dA_offset, ldda,
            hA_dst, hA_offset, ldha,
            queue, event);
        return err;
    }
}

// --------------------
magma_err_t
magma_dsetvector_async(
    magma_int_t n,
    double const* hA_src, size_t hA_offset, magma_int_t incx,
    magmaDouble_ptr dA_dst, size_t dA_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event )
{
    if ( n <= 0 )
       return MAGMA_SUCCESS;

    cl_int err;
    if(incx == 1 && incy == 1) {
        err = clEnqueueWriteBuffer(
            queue, dA_dst, CL_FALSE,
            dA_offset*sizeof(double), n*sizeof(double),
            hA_src+hA_offset, 0, NULL, event);
        return err;
    } else {
        magma_int_t ldha = incx;
        magma_int_t ldda = incy;
        cl_int err = magma_dsetmatrix_async(1, n,
            hA_src, hA_offset, ldha,
            dA_dst, dA_offset, ldda,
            queue, event);
        return err;
    }
}

// --------------------
magma_err_t
magma_dsetmatrix_async(
    magma_int_t m, magma_int_t n,
    double const* hA_src, size_t hA_offset, magma_int_t ldha,
    magmaDouble_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue, magma_event_t *event )
{
    if ( m<=0 || n <= 0 )
       return MAGMA_SUCCESS;

    size_t buffer_origin[3] = { dA_offset*sizeof(double), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(double), 0, 0 };
    size_t region[3]        = { m*sizeof(double), n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
        queue, dA_dst, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(double), 0,
        ldha*sizeof(double), 0,
        hA_src, 0, NULL, event );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_dgetmatrix_async(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    double*          hA_dst, size_t hA_offset, magma_int_t ldha,
    magma_queue_t queue, magma_event_t *event )
{
    if (m<=0 || n <= 0)
      return MAGMA_SUCCESS;

    size_t buffer_origin[3] = { dA_offset*sizeof(double), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(double), 0, 0 };
    size_t region[3]        = { m*sizeof(double), n, 1 };
    cl_int err = clEnqueueReadBufferRect(
        queue, dA_src, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(double), 0,
        ldha*sizeof(double), 0,
        hA_dst, 0, NULL, event );
    clFlush(queue);
    return err;
}

// --------------------
magma_err_t
magma_dcopymatrix(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    magmaDouble_ptr    dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
{
    if ( m<=0 || n <= 0 )
       return MAGMA_SUCCESS;

    size_t src_origin[3] = { dA_offset*sizeof(double), 0, 0 };
    size_t dst_orig[3]   = { dB_offset*sizeof(double), 0, 0 };
    size_t region[3]        = { m*sizeof(double), n, 1 };
    cl_int err = clEnqueueCopyBufferRect(
        queue, dA_src, dB_dst,
        src_origin, dst_orig, region,
        ldda*sizeof(double), 0,
        lddb*sizeof(double), 0,
        0, NULL, gevent );
    return err;
}
// ========================================
// BLAS functions
magma_err_t
magma_dgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    double beta,  magmaDouble_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0 || k<=0)
      return MAGMA_SUCCESS;

    cl_int err = clAmdBlasDgemmEx(
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
magma_dgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dx, size_t dx_offset, magma_int_t incx,
    double beta,  magmaDouble_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasDgemvEx(
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
magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    double beta,  magmaDouble_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasDsymm(
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
magma_dsymv(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dx, size_t dx_offset, magma_int_t incx,
    double beta,  magmaDouble_ptr       dy, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if ( n <= 0 )
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasDsymvEx(
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
magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
    double beta,  magmaDouble_ptr       dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue )
{
    if (n<=0 || k <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasDsyrkEx(
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
magma_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasDtrsmEx(
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
magma_dtrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
    magmaDouble_ptr dx, size_t dx_offset, magma_int_t incx,
    magma_queue_t queue )
{
    if ( n <= 0 )
        return MAGMA_SUCCESS;

    cl_int err = clAmdBlasDtrsv(
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
magma_dtrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_ptr       dB, size_t dB_offset, magma_int_t ldb,
    magma_queue_t queue )
{
    if (m<=0 || n <= 0)
       return MAGMA_SUCCESS;

    cl_int err = clAmdBlasDtrmmEx(
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
magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t lda,
                              magmaDouble_const_ptr dB, size_t dB_offset, magma_int_t ldb,
    double beta, magmaDouble_ptr dC, size_t dC_offset, magma_int_t ldc,
    magma_queue_t queue)
{
     if (n<=0 || k <= 0)
        return MAGMA_SUCCESS;

     cl_int err = clAmdBlasDsyr2kEx(
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
