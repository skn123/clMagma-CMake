/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @author Stan Tomov
       @precisions normal z -> s d c
*/
#include "common_magma.h"
#define PRECISION_z
#include "magmablas.h"

//===========================================================================
//  Set a matrix from CPU to multi-GPUs is 1D block cyclic distribution.
//  The dA arrays are pointers to the matrix data for the corresponding GPUs.
//===========================================================================
extern "C" void
magmablas_zsetmatrix_1D_bcyclic( magma_int_t m, magma_int_t n,
                                 const magmaDoubleComplex *hA,   magma_int_t lda,
                                 magmaDoubleComplex_ptr *dA, magma_int_t ldda,
                                 magma_int_t num_gpus, magma_int_t nb, 
                                 magma_queue_t* trans_queues)
{
    magma_int_t i, d, nk;

    for( i = 0; i < n; i += nb ) {
        d = (i/nb) % num_gpus;
        nk = min(nb, n-i);
        magma_zsetmatrix( m, nk, 
                        &hA[i*lda], 0, lda, 
                        dA[d], i/(nb*num_gpus)*nb*ldda, ldda, 
                        trans_queues[d]);
    }
}


//===========================================================================
//  Get a matrix with 1D block cyclic distribution on multiGPUs to the CPU.
//  The dA arrays are pointers to the matrix data for the corresponding GPUs.
//===========================================================================
extern "C" void
magmablas_zgetmatrix_1D_bcyclic( magma_int_t m, magma_int_t n,
                                 magmaDoubleComplex_ptr *dA, magma_int_t ldda,
                                 magmaDoubleComplex *hA, magma_int_t lda,
                                 magma_int_t num_gpus, magma_int_t nb, 
                                 magma_queue_t* trans_queues)
{
    magma_int_t i, d, nk;

    for( i = 0; i < n; i += nb ) {
        d = (i/nb) % num_gpus;
        nk = min(nb, n-i);
        magma_zgetmatrix( m, nk, 
                        dA[d], i/(nb*num_gpus)*nb*ldda, ldda, 
                        &hA[i*lda], 0, lda, 
                        trans_queues[d]);
    }
}
