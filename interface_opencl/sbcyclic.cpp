/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @author Stan Tomov
       @generated from zbcyclic.cpp normal z -> s, Fri Jan 10 15:51:19 2014
*/
#include "common_magma.h"
#define PRECISION_s
#include "magmablas.h"

//===========================================================================
//  Set a matrix from CPU to multi-GPUs is 1D block cyclic distribution.
//  The dA arrays are pointers to the matrix data for the corresponding GPUs.
//===========================================================================
extern "C" void
magmablas_ssetmatrix_1D_bcyclic( magma_int_t m, magma_int_t n,
                                 const float *hA,   magma_int_t lda,
                                 magmaFloat_ptr *dA, magma_int_t ldda,
                                 magma_int_t num_gpus, magma_int_t nb, 
                                 magma_queue_t* trans_queues)
{
    magma_int_t i, d, nk;

    for( i = 0; i < n; i += nb ) {
        d = (i/nb) % num_gpus;
        nk = min(nb, n-i);
        magma_ssetmatrix( m, nk, 
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
magmablas_sgetmatrix_1D_bcyclic( magma_int_t m, magma_int_t n,
                                 magmaFloat_ptr *dA, magma_int_t ldda,
                                 float *hA, magma_int_t lda,
                                 magma_int_t num_gpus, magma_int_t nb, 
                                 magma_queue_t* trans_queues)
{
    magma_int_t i, d, nk;

    for( i = 0; i < n; i += nb ) {
        d = (i/nb) % num_gpus;
        nk = min(nb, n-i);
        magma_sgetmatrix( m, nk, 
                        dA[d], i/(nb*num_gpus)*nb*ldda, ldda, 
                        &hA[i*lda], 0, lda, 
                        trans_queues[d]);
    }
}
