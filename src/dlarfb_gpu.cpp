/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zlarfb_gpu.cpp normal z -> d, Fri Jan 10 15:51:18 2014
*/

#include <stdio.h>
#include "common_magma.h"

magma_err_t
magma_dlarfb_gpu( int side, int trans, int direct, int storev,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  magmaDouble_ptr dV, size_t dV_offset,   magma_int_t ldv,
                  magmaDouble_ptr dT, size_t dT_offset,   magma_int_t ldt,
                  magmaDouble_ptr dC, size_t dC_offset,   magma_int_t ldc,
                  magmaDouble_ptr dwork, size_t dwork_offset, magma_int_t ldwork,
                  magma_queue_t queue)
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose
    =======
    DLARFB applies a real block reflector H or its transpose H' to a
    DOUBLE_PRECISION m by n matrix C, from the left.

    Arguments
    =========
    SIDE    (input) CHARACTER
            = 'L': apply H or H' from the Left
            = 'R': apply H or H' from the Right

    TRANS   (input) CHARACTER
            = 'N': apply H  (No transpose)
            = 'C': apply H' (Conjugate transpose)

    DIRECT  (input) CHARACTER
            Indicates how H is formed from a product of elementary
            reflectors
            = 'F': H = H(1) H(2) . . . H(k) (Forward)
            = 'B': H = H(k) . . . H(2) H(1) (Backward)

    STOREV  (input) CHARACTER
            Indicates how the vectors which define the elementary
            reflectors are stored:
            = 'C': Columnwise
            = 'R': Rowwise

    M       (input) INTEGER
            The number of rows of the matrix C.

    N       (input) INTEGER
            The number of columns of the matrix C.

    K       (input) INTEGER
            The order of the matrix T (= the number of elementary
            reflectors whose product defines the block reflector).

    DV      (input) DOUBLE_PRECISION array, dimension (LDV,K)
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V. LDV >= max(1,M);

    DT      (input) DOUBLE_PRECISION array, dimension (LDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    DC      (input/output) DOUBLE_PRECISION array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) DOUBLE_PRECISION array, dimension (LDWORK,K)

    LDWORK  (input) INTEGER
            The leading dimension of the array WORK.
            If SIDE == 'L', LDWORK >= max(1,N);
            if SIDE == 'R', LDWORK >= max(1,M);
    ===================================================================      */

    /* TODO: replace with updated larfb_gpu from CUDA MAGMA */
    
#define dV(i)       dV, (i)
#define dT(i)       dT, (i)
#define dC(i)       dC, (i)
#define dwork(i) dwork, (i)

    double c_zero    = MAGMA_D_MAKE(  0.0, 0.0 );
    double c_one     = MAGMA_D_MAKE(  1.0, 0.0 );
    double c_neg_one = MAGMA_D_MAKE( -1.0, 0.0 );

    if (m <= 0 || n <= 0) {
        return MAGMA_SUCCESS;
    }

    magma_int_t transt;
    if (trans == MagmaNoTrans)
      transt = MagmaTrans;
    else
      transt = MagmaNoTrans;

    if ( side  == MagmaLeft ) {

    if ( storev == MagmaColumnwise )
      {
        magma_dgemm( MagmaTrans, MagmaNoTrans,
                     n, k, m,
                     c_one,  dC(dC_offset),    ldc,
                     dV(dV_offset),    ldv,
                     c_zero, dwork(dwork_offset), ldwork, queue);

        if (direct == MagmaForward)
            magma_dtrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                         n, k,
                         c_one, dT(dT_offset),    ldt,
                         dwork(dwork_offset), ldwork, queue);
        else
            magma_dtrmm( MagmaRight, MagmaLower, transt, MagmaNonUnit,
                         n, k,
                         c_one, dT(dT_offset),    ldt,
                         dwork(dwork_offset), ldwork, queue);

        magma_dgemm( MagmaNoTrans, MagmaTrans,
                     m, n, k,
                     c_neg_one, dV(dV_offset),    ldv,
                     dwork(dwork_offset), ldwork,
                     c_one,     dC(dC_offset),    ldc, queue);
    }
    else {
        magma_dgemm( MagmaNoTrans, MagmaTrans,
                     m, k, n,
                     c_one,  dC(dC_offset),    ldc,
                     dV(dV_offset),    ldv,
                     c_zero, dwork(dwork_offset), ldwork, queue);

        magma_dtrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                     m, k,
                     c_one, dT(dT_offset),    ldt,
                     dwork(dwork_offset), ldwork, queue);
        
        magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                     m, n, k,
                     c_neg_one, dwork(dwork_offset), ldwork,
                     dV(dV_offset),    ldv,
                     c_one,     dC(dC_offset),    ldc, queue);
    }
    }
    
    else {

        /* Case side == 'R' */
        if ( storev == MagmaColumnwise ) {
            magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                         m, k, n,
                         c_one,  dC(dC_offset),    ldc,
                         dV(dV_offset),    ldv,
                         c_zero, dwork(dwork_offset), ldwork, queue);
            // ??? ldwork replaced by k for case n < k

            if (direct == MagmaForward)
                magma_dtrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                             m, k,
                             c_one, dT(dT_offset),    ldt,
                             dwork(dwork_offset), ldwork, queue);
            else
                magma_dtrmm( MagmaRight, MagmaLower, transt, MagmaNonUnit,
                             m, k,
                             c_one, dT(dT_offset),    ldt,
                             dwork(dwork_offset), ldwork, queue);

            magma_dgemm( MagmaNoTrans, MagmaTrans,
                         m, n, k,
                         c_neg_one, dwork(dwork_offset), ldwork,
                         dV(dV_offset),    ldv,
                         c_one,     dC(dC_offset),    ldc, queue);
        }
        else {
            magma_dgemm( MagmaNoTrans, MagmaTrans,
                         m, k, n,
                         c_one,  dC(dC_offset),    ldc,
                         dV(dV_offset),    ldv,
                         c_zero, dwork(dwork_offset), ldwork, queue);

            magma_dtrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                         m, k,
                         c_one, dT(dT_offset),    ldt,
                         dwork(dwork_offset), ldwork, queue);

            magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                         m, n, k,
                         c_neg_one, dwork(dwork_offset), ldwork,
                         dV(dV_offset),    ldv,
                         c_one,     dC(dC_offset),    ldc, queue);
        }
    }
    
    return MAGMA_SUCCESS;
} /* magma_dlarfb */
