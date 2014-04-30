/*
    -- MAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @author Mark Gates
       @generated from zprint.cpp normal z -> s, Fri Jan 10 15:51:17 2014

*/
#include "common_magma.h"

#define A(i,j) (A + (i) + (j)*lda)

// -------------------------
// Prints a matrix that is on the CPU host.
extern "C"
void magma_sprint( int m, int n, float *A, int lda )
{
    float c_zero = MAGMA_S_ZERO;
    
    printf( "[\n" );
    for( int i = 0; i < m; ++i ) {
        for( int j = 0; j < n; ++j ) {
            if ( MAGMA_S_EQUAL( *A(i,j), c_zero )) {
                printf( "   0.    " );
            }
            else {
                printf( " %8.4f", MAGMA_S_REAL( *A(i,j) ));
            }
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

// -------------------------
// Prints a matrix that is on the GPU device.
// Internally allocates memory on host, copies it to the host, prints it,
// and de-allocates host memory.
extern "C"
void magma_sprint_gpu( int m, int n, magmaFloat_ptr dA, size_t dA_offset, int ldda, magma_queue_t queue )
{
    int lda = m;
    float* A = (float*) malloc( lda*n*sizeof(float) );
    magma_sgetmatrix( m, n, dA, dA_offset, ldda,  A, 0, lda, queue );
    
    magma_sprint( m, n, A, lda );
    
    free( A );
}
