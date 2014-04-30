/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from zlarfb_gpu.cpp normal z -> c, Fri Jan 10 15:51:18 2014
*/

#include <stdio.h>
#include "common_magma.h"

magma_err_t
magma_clarfb_gpu( int side, int trans, int direct, int storev,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  magmaFloatComplex_ptr dV, size_t dV_offset,   magma_int_t ldv,
                  magmaFloatComplex_ptr dT, size_t dT_offset,   magma_int_t ldt,
                  magmaFloatComplex_ptr dC, size_t dC_offset,   magma_int_t ldc,
                  magmaFloatComplex_ptr dwork, size_t dwork_offset, magma_int_t ldwork,
                  magma_queue_t queue)
{
/*  -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    Purpose
    =======
    CLARFB applies a complex block reflector H or its transpose H' to a
    COMPLEX m by n matrix C, from the left.

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

    DV      (input) COMPLEX array, dimension (LDV,K)
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V. LDV >= max(1,M);

    DT      (input) COMPLEX array, dimension (LDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    DC      (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) COMPLEX array, dimension (LDWORK,K)

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

    magmaFloatComplex c_zero    = MAGMA_C_MAKE(  0.0, 0.0 );
    magmaFloatComplex c_one     = MAGMA_C_MAKE(  1.0, 0.0 );
    magmaFloatComplex c_neg_one = MAGMA_C_MAKE( -1.0, 0.0 );

    if (m <= 0 || n <= 0) {
        return MAGMA_SUCCESS;
    }

    magma_int_t transt;
    if (trans == MagmaNoTrans)
      transt = MagmaConjTrans;
    else
      transt = MagmaNoTrans;

    if ( side  == MagmaLeft ) {

    if ( storev == MagmaColumnwise )
      {
        magma_cgemm( MagmaConjTrans, MagmaNoTrans,
                     n, k, m,
                     c_one,  dC(dC_offset),    ldc,
                     dV(dV_offset),    ldv,
                     c_zero, dwork(dwork_offset), ldwork, queue);

        if (direct == MagmaForward)
            magma_ctrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                         n, k,
                         c_one, dT(dT_offset),    ldt,
                         dwork(dwork_offset), ldwork, queue);
        else
            magma_ctrmm( MagmaRight, MagmaLower, transt, MagmaNonUnit,
                         n, k,
                         c_one, dT(dT_offset),    ldt,
                         dwork(dwork_offset), ldwork, queue);

        magma_cgemm( MagmaNoTrans, MagmaConjTrans,
                     m, n, k,
                     c_neg_one, dV(dV_offset),    ldv,
                     dwork(dwork_offset), ldwork,
                     c_one,     dC(dC_offset),    ldc, queue);
    }
    else {
        magma_cgemm( MagmaNoTrans, MagmaConjTrans,
                     m, k, n,
                     c_one,  dC(dC_offset),    ldc,
                     dV(dV_offset),    ldv,
                     c_zero, dwork(dwork_offset), ldwork, queue);

        magma_ctrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                     m, k,
                     c_one, dT(dT_offset),    ldt,
                     dwork(dwork_offset), ldwork, queue);
        
        magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                     m, n, k,
                     c_neg_one, dwork(dwork_offset), ldwork,
                     dV(dV_offset),    ldv,
                     c_one,     dC(dC_offset),    ldc, queue);
    }
    }
    
    else {

        /* Case side == 'R' */
        if ( storev == MagmaColumnwise ) {
            magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                         m, k, n,
                         c_one,  dC(dC_offset),    ldc,
                         dV(dV_offset),    ldv,
                         c_zero, dwork(dwork_offset), ldwork, queue);
            // ??? ldwork replaced by k for case n < k

            if (direct == MagmaForward)
                magma_ctrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                             m, k,
                             c_one, dT(dT_offset),    ldt,
                             dwork(dwork_offset), ldwork, queue);
            else
                magma_ctrmm( MagmaRight, MagmaLower, transt, MagmaNonUnit,
                             m, k,
                             c_one, dT(dT_offset),    ldt,
                             dwork(dwork_offset), ldwork, queue);

            magma_cgemm( MagmaNoTrans, MagmaConjTrans,
                         m, n, k,
                         c_neg_one, dwork(dwork_offset), ldwork,
                         dV(dV_offset),    ldv,
                         c_one,     dC(dC_offset),    ldc, queue);
        }
        else {
            magma_cgemm( MagmaNoTrans, MagmaConjTrans,
                         m, k, n,
                         c_one,  dC(dC_offset),    ldc,
                         dV(dV_offset),    ldv,
                         c_zero, dwork(dwork_offset), ldwork, queue);

            magma_ctrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                         m, k,
                         c_one, dT(dT_offset),    ldt,
                         dwork(dwork_offset), ldwork, queue);

            magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                         m, n, k,
                         c_neg_one, dwork(dwork_offset), ldwork,
                         dV(dV_offset),    ldv,
                         c_one,     dC(dC_offset),    ldc, queue);
        }
    }
    
    return MAGMA_SUCCESS;
} /* magma_clarfb */
