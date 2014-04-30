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

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"

#define PRECISION_z
#ifdef HAVE_clAmdBlas

// ========================================
// globals, defined in interface.c
extern cl_platform_id gPlatform;
extern cl_context     gContext;


// ========================================
// copying sub-matrices (contiguous columns)
// OpenCL takes queue even for blocking transfers, oddly.
    magma_err_t
magma_zsetmatrix(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex const* hA_src, size_t hA_offset, magma_int_t ldha,
        magmaDoubleComplex_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
        magma_queue_t queue )
{
    size_t buffer_origin[3] = { dA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t region[3]        = { m*sizeof(magmaDoubleComplex), n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
            queue, dA_dst, CL_TRUE,  // blocking
            buffer_origin, host_orig, region,
            ldda*sizeof(magmaDoubleComplex), 0,
            ldha*sizeof(magmaDoubleComplex), 0,
            hA_src, 0, NULL, NULL );
    return err;
}

    magma_err_t
magma_zsetmatrix_trace(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex const* hA_src, size_t hA_offset, magma_int_t ldha,
        magmaDoubleComplex_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
        magma_queue_t queue, magma_event_t *event )
{
    size_t buffer_origin[3] = { dA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t region[3]        = { m*sizeof(magmaDoubleComplex), n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
            queue, dA_dst, CL_TRUE,  // blocking
            buffer_origin, host_orig, region,
            ldda*sizeof(magmaDoubleComplex), 0,
            ldha*sizeof(magmaDoubleComplex), 0,
            hA_src, 0, NULL, event );
    return err;
}

// --------------------
    magma_err_t 
magma_zsetvector(
        magma_int_t n, 
        magmaDoubleComplex const* hA_src, size_t hA_offset, magma_int_t incx, 
        magmaDoubleComplex_ptr dA_dst, size_t dA_offset, magma_int_t incy, 
        magma_queue_t queue )
{
    cl_int err;
    if(incx == 1 && incy == 1){
        err = clEnqueueWriteBuffer(
                queue, dA_dst, CL_TRUE, 
                dA_offset*sizeof(magmaDoubleComplex), n*sizeof(magmaDoubleComplex), 
                hA_src+hA_offset, 0, NULL, NULL);
        return err;
    }else{
        magma_int_t ldha = incx;
        magma_int_t ldda = incy;
        cl_int err = magma_zsetmatrix(1, n, 
                hA_src, hA_offset, ldha, 
                dA_dst, dA_offset, ldda, 
                queue);
        return err;
    }
}

// --------------------
    magma_err_t
magma_zgetmatrix(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
        magmaDoubleComplex*          hA_dst, size_t hA_offset, magma_int_t ldha,
        magma_queue_t queue )
{
    size_t buffer_origin[3] = { dA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t region[3]        = { m*sizeof(magmaDoubleComplex), n, 1 };
    cl_int err = clEnqueueReadBufferRect(
            queue, dA_src, CL_TRUE,  // blocking
            buffer_origin, host_orig, region,
            ldda*sizeof(magmaDoubleComplex), 0,
            ldha*sizeof(magmaDoubleComplex), 0,
            hA_dst, 0, NULL, NULL );
    return err;
}

    magma_err_t
magma_zgetmatrix_trace(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
        magmaDoubleComplex*          hA_dst, size_t hA_offset, magma_int_t ldha,
        magma_queue_t queue, magma_event_t *event )
{
    size_t buffer_origin[3] = { dA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t region[3]        = { m*sizeof(magmaDoubleComplex), n, 1 };
    cl_int err = clEnqueueReadBufferRect(
            queue, dA_src, CL_TRUE,  // blocking
            buffer_origin, host_orig, region,
            ldda*sizeof(magmaDoubleComplex), 0,
            ldha*sizeof(magmaDoubleComplex), 0,
            hA_dst, 0, NULL, event );
    return err;
}

// --------------------
    magma_err_t 
magma_zgetvector(
        magma_int_t n, 
        magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t incx, 
        magmaDoubleComplex*             hA_dst, size_t hA_offset, magma_int_t incy,
        magma_queue_t queue )
{
    cl_int err;
    if(incx ==1 && incy ==1){
        err = clEnqueueReadBuffer(
                queue, dA_src, CL_TRUE, 
                dA_offset*sizeof(magmaDoubleComplex), n*sizeof(magmaDoubleComplex), 
                hA_dst+hA_offset, 0, NULL, NULL);
        return err;            
    }else{
        magma_int_t ldda = incx;
        magma_int_t ldha = incy;
        err = magma_zgetmatrix(1, n, 
                dA_src, dA_offset, ldda,
                hA_dst, hA_offset, ldha,
                queue);
        return err;
    }
}

// --------------------
    magma_err_t
magma_zsetmatrix_async(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex const* hA_src, size_t hA_offset, magma_int_t ldha,
        magmaDoubleComplex_ptr    dA_dst, size_t dA_offset, magma_int_t ldda,
        magma_queue_t queue, magma_event_t *event )
{
    size_t buffer_origin[3] = { dA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t region[3]        = { m*sizeof(magmaDoubleComplex), n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
            queue, dA_dst, CL_FALSE,  // non-blocking
            buffer_origin, host_orig, region,
            ldda*sizeof(magmaDoubleComplex), 0,
            ldha*sizeof(magmaDoubleComplex), 0,
            hA_src, 0, NULL, event );
    return err;
}

// --------------------
    magma_err_t
magma_zgetmatrix_async(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
        magmaDoubleComplex*          hA_dst, size_t hA_offset, magma_int_t ldha,
        magma_queue_t queue, magma_event_t *event )
{
    size_t buffer_origin[3] = { dA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t host_orig[3]     = { hA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t region[3]        = { m*sizeof(magmaDoubleComplex), n, 1 };
    cl_int err = clEnqueueReadBufferRect(
            queue, dA_src, CL_FALSE,  // non-blocking
            buffer_origin, host_orig, region,
            ldda*sizeof(magmaDoubleComplex), 0,
            ldha*sizeof(magmaDoubleComplex), 0,
            hA_dst, 0, NULL, event );
    return err;
}

// --------------------
    magma_err_t
magma_zcopymatrix(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
        magmaDoubleComplex_ptr    dB_dst, size_t dB_offset, magma_int_t lddb,
        magma_queue_t queue )
{
    size_t src_origin[3] = { dA_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t dst_orig[3]   = { dB_offset*sizeof(magmaDoubleComplex), 0, 0 };
    size_t region[3]        = { m*sizeof(magmaDoubleComplex), n, 1 };
    cl_int err = clEnqueueCopyBufferRect(
            queue, dA_src, dB_dst,
            src_origin, dst_orig, region,
            ldda*sizeof(magmaDoubleComplex), 0,
            lddb*sizeof(magmaDoubleComplex), 0,
            0, NULL, NULL );
    return err;
}
// ========================================
// BLAS functions
    magma_err_t
magma_zgemm(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
        magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue )
{
    cl_int err = clAmdBlasZgemmEx(
            clAmdBlasColumnMajor,
            amdblas_trans_const( transA ),
            amdblas_trans_const( transB ),
            m, n, k,
            alpha, dA, dA_offset, lda,
            dB, dB_offset, ldb,
            beta,  dC, dC_offset, ldc,
            1, &queue, 0, NULL, NULL );
    clFlush(queue);
    return err;
}

    magma_err_t
magma_zgemm_trace(
        magma_trans_t transA, magma_trans_t transB,
        magma_int_t m, magma_int_t n, magma_int_t k,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
        magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue, magma_event_t* event )
{
    cl_int err = clAmdBlasZgemmEx(
            clAmdBlasColumnMajor,
            amdblas_trans_const( transA ),
            amdblas_trans_const( transB ),
            m, n, k,
            alpha, dA, dA_offset, lda,
            dB, dB_offset, ldb,
            beta,  dC, dC_offset, ldc,
            1, &queue, 0, NULL, event );
    clFlush(queue);
    return err;
}

// --------------------
    magma_err_t
magma_zgemv(
        magma_trans_t transA,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
        magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
        magma_queue_t queue )
{
    cl_int err = clAmdBlasZgemvEx(
            clAmdBlasColumnMajor,
            amdblas_trans_const( transA ),
            m, n,
            alpha, dA, dA_offset, lda,
            dx, dx_offset, incx,
            beta,  dy, dy_offset, incy,
            1, &queue, 0, NULL, NULL );
    return err;
}

// --------------------
    magma_err_t
magma_zhemm(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb,
        magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue )
{
    cl_int err = clAmdBlasZhemm(
            clAmdBlasColumnMajor,
            amdblas_side_const( side ),
            amdblas_uplo_const( uplo ),
            m, n,
            alpha, dA, dA_offset, lda,
            dB, dB_offset, ldb,
            beta,  dC, dC_offset, ldc,
            1, &queue, 0, NULL, NULL );
    return err;
}

// --------------------
    magma_err_t
magma_zhemv(
        magma_uplo_t uplo,
        magma_int_t n,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_const_ptr dx, size_t dx_offset, magma_int_t incx,
        magmaDoubleComplex beta,  magmaDoubleComplex_ptr       dy, size_t dy_offset, magma_int_t incy,
        magma_queue_t queue )
{
    cl_int err = clAmdBlasZhemv(
            clAmdBlasColumnMajor,
            amdblas_uplo_const( uplo ),
            n,
            alpha, dA, dA_offset, lda,
            dx, dx_offset, incx,
            beta,  dy, dy_offset, incy,
            1, &queue, 0, NULL, NULL );
    return err;
}

// --------------------
    magma_err_t
magma_zherk(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t n, magma_int_t k,
        double alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        double beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue )
{
    cl_int err = clAmdBlasZherk(
            clAmdBlasColumnMajor,
            amdblas_uplo_const( uplo ),
            amdblas_trans_const( trans ),
            n, k,
            alpha, dA, dA_offset, lda,
            beta,  dC, dC_offset, ldc,
            1, &queue, 0, NULL, NULL );
    clFlush(queue);
    return err;
}

    magma_err_t
magma_zherk_trace(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t n, magma_int_t k,
        double alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        double beta,  magmaDoubleComplex_ptr       dC, size_t dC_offset, magma_int_t ldc,
        magma_queue_t queue, magma_event_t* event )
{
    cl_int err = clAmdBlasZherk(
            clAmdBlasColumnMajor,
            amdblas_uplo_const( uplo ),
            amdblas_trans_const( trans ),
            n, k,
            alpha, dA, dA_offset, lda,
            beta,  dC, dC_offset, ldc,
            1, &queue, 0, NULL, event );
    clFlush(queue);
    return err;
}

// --------------------
    magma_err_t
magma_ztrsm(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
        magma_queue_t queue )
{
/*    
    magmaDoubleComplex *hA, *hB;
    if(side==MagmaRight){
        hA = (magmaDoubleComplex*)malloc(lda*n*sizeof(magmaDoubleComplex));
        hB = (magmaDoubleComplex*)malloc(ldb*n*sizeof(magmaDoubleComplex));
        magma_zgetmatrix(n, n, dA, dA_offset, lda, hA, 0, lda, queue);
        magma_zgetmatrix(m, n, dB, dB_offset, ldb, hB, 0, ldb, queue);
#if defined(PRECISION_z) || defined(PRECISION_c)
        cblas_ztrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans, (CBLAS_DIAG)diag,
                m, n,
                &alpha, hA, lda, hB, ldb);
#else
        cblas_ztrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans, (CBLAS_DIAG)diag,
                m, n,
                alpha, hA, lda, hB, ldb);
#endif
        magma_zsetmatrix(m, n, hB, 0, ldb, dB, dB_offset, ldb, queue);
        free(hB);
        free(hA);
    }else{
        hA = (magmaDoubleComplex*)malloc(lda*m*sizeof(magmaDoubleComplex));
        hB = (magmaDoubleComplex*)malloc(ldb*n*sizeof(magmaDoubleComplex));
        magma_zgetmatrix(m, m, dA, dA_offset, lda, hA, 0, lda, queue);
        magma_zgetmatrix(m, n, dB, dB_offset, ldb, hB, 0, ldb, queue);
#if defined(PRECISION_z) || defined(PRECISION_c)
        cblas_ztrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans, (CBLAS_DIAG)diag, 
                m, n,
                &alpha, hA, lda, hB, ldb);
#else
        cblas_ztrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans, (CBLAS_DIAG)diag, 
                m, n,
                alpha, hA, lda, hB, ldb);
#endif
        magma_zsetmatrix(m, n, hB, 0, ldb, dB, dB_offset, ldb, queue);
        free(hB);
        free(hA);
    }
    return CL_SUCCESS;
*/
       cl_int err = clAmdBlasZtrsmEx(
       clAmdBlasColumnMajor,
       amdblas_side_const( side ),
       amdblas_uplo_const( uplo ),
       amdblas_trans_const( trans ),
       amdblas_diag_const( diag ),
       m, n,
       alpha, dA, dA_offset, lda,
       dB, dB_offset, ldb,
       1, &queue, 0, NULL, NULL );
       clFlush(queue);
       return err;
}

    magma_err_t
magma_ztrsm_trace(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
        magma_queue_t queue, magma_event_t* event )
{
    cl_int err = clAmdBlasZtrsmEx(
            clAmdBlasColumnMajor,
            amdblas_side_const( side ),
            amdblas_uplo_const( uplo ),
            amdblas_trans_const( trans ),
            amdblas_diag_const( diag ),
            m, n,
            alpha, dA, dA_offset, lda,
            dB, dB_offset, ldb,
            1, &queue, 0, NULL, event );
    clFlush(queue);
    return err;
}

// --------------------
    magma_err_t
magma_ztrsv(
        magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
        magma_int_t n, 
        magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_ptr dx, size_t dx_offset, magma_int_t incx,
        magma_queue_t queue )
{
    cl_int err = clAmdBlasZtrsv(
            clAmdBlasColumnMajor,
            amdblas_uplo_const( uplo ),
            amdblas_trans_const( trans ),
            amdblas_diag_const( diag ),
            n,
            dA, dA_offset, lda,
            dx, dx_offset, incx,
            1, &queue, 0, NULL, NULL );
    return err;
}

// --------------------
    magma_err_t
magma_ztrmm(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda,
        magmaDoubleComplex_ptr       dB, size_t dB_offset, magma_int_t ldb,
        magma_queue_t queue )
{
    cl_int err = clAmdBlasZtrmmEx(
            clAmdBlasColumnMajor,
            amdblas_side_const( side ),
            amdblas_uplo_const( uplo ),
            amdblas_trans_const( trans ),
            amdblas_diag_const( diag ),
            m, n,
            alpha, dA, dA_offset, lda,
            dB, dB_offset, ldb,
            1, &queue, 0, NULL, NULL );
    clFlush(queue);
    return err;
}

// --------------------
    magma_err_t
magma_zher2k(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t n, magma_int_t k, 
        magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, size_t dA_offset, magma_int_t lda, 
        magmaDoubleComplex_const_ptr dB, size_t dB_offset, magma_int_t ldb, 
        double beta, magmaDoubleComplex_ptr dC, size_t dC_offset, magma_int_t ldc, 
        magma_queue_t queue)
{    // cblas wrapper
    magma_int_t ka, kb;
    if(trans == MagmaNoTrans){
        ka = k;
        kb = k;
    }else{
        ka = n;
        kb = n;
    }
    magmaDoubleComplex *hA, *hB, *hC;
    hA = (magmaDoubleComplex*)malloc(lda*ka*sizeof(magmaDoubleComplex));
    hB = (magmaDoubleComplex*)malloc(ldb*kb*sizeof(magmaDoubleComplex));
    hC = (magmaDoubleComplex*)malloc(ldc*n*sizeof(magmaDoubleComplex));
    magma_zgetmatrix(lda, ka, dA, dA_offset, lda, hA, 0, lda, queue);
    magma_zgetmatrix(ldb, kb, dB, dB_offset, ldb, hB, 0, ldb, queue);
    magma_zgetmatrix(ldc, n, dC, dC_offset, ldc, hC, 0, ldc, queue);
#if defined(PRECISION_z) || defined(PRECISION_c)
    cblas_zher2k(CblasColMajor, cblas_uplo_const(uplo), cblas_trans_const(trans), 
            n, k, (void*)&alpha, hA, lda, hB, ldb, beta, hC, ldc);
#else
    cblas_zher2k(CblasColMajor, cblas_uplo_const(uplo), cblas_trans_const(trans), 
            n, k, alpha, hA, lda, hB, ldb, beta, hC, ldc);
#endif
    magma_zsetmatrix(ldc, n, hC, 0, ldc, dC, dC_offset, ldc, queue);    
    free(hA);
    free(hB);
    free(hC);
    return CL_SUCCESS;
}

#endif // HAVE_clAmdBlas
