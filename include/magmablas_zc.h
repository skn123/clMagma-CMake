/*
    -- MAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions mixed zc -> ds
*/

#ifndef MAGMA_BLAS_ZC_H
#define MAGMA_BLAS_ZC_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */

magma_err_t
magmablas_zlag2c( magma_int_t M, magma_int_t N , 
                  magmaDoubleComplex_ptr A, size_t A_offset,
                  magma_int_t lda, 
                  magmaFloatComplex_ptr SA, size_t SA_offset, 
                  magma_int_t ldsa, 
                  magma_int_t *info, magma_queue_t queue );

magma_err_t
magmablas_clag2z( magma_int_t m, magma_int_t n , 
                  magmaDoubleComplex_ptr SA, size_t SA_offset,
                  magma_int_t ldsa, 
                  magmaFloatComplex_ptr A, size_t A_offset, 
                  magma_int_t lda, 
                  magma_int_t *info, magma_queue_t queue ) ;


#ifdef __cplusplus
}
#endif

#endif /* MAGMA_BLAS_ZC_H */
