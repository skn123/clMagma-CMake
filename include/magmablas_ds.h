/*
    -- MAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from magmablas_zc.h mixed zc -> ds, Fri Jan 10 15:51:16 2014
*/

#ifndef MAGMA_BLAS_DS_H
#define MAGMA_BLAS_DS_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */

magma_err_t
magmablas_dlag2s( magma_int_t M, magma_int_t N , 
                  magmaDouble_ptr A, size_t A_offset,
                  magma_int_t lda, 
                  magmaFloat_ptr SA, size_t SA_offset, 
                  magma_int_t ldsa, 
                  magma_int_t *info, magma_queue_t queue );

magma_err_t
magmablas_slag2d( magma_int_t m, magma_int_t n , 
                  magmaDouble_ptr SA, size_t SA_offset,
                  magma_int_t ldsa, 
                  magmaFloat_ptr A, size_t A_offset, 
                  magma_int_t lda, 
                  magma_int_t *info, magma_queue_t queue ) ;


#ifdef __cplusplus
}
#endif

#endif /* MAGMA_BLAS_DS_H */
