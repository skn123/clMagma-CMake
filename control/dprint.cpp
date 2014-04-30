/*
    -- MAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @author Mark Gates
       @generated from zprint.cpp normal z -> d, Fri Jan 10 15:51:17 2014

*/
#include "common_magma.h"

#define A(i,j) (A + (i) + (j)*lda)

// -------------------------
// Prints a matrix that is on the CPU host.
extern "C"
void magma_dprint( int m, int n, double *A, int lda )
{
    double c_zero = MAGMA_D_ZERO;
    
    printf( "[\n" );
    for( int i = 0; i < m; ++i ) {
        for( int j = 0; j < n; ++j ) {
            if ( MAGMA_D_EQUAL( *A(i,j), c_zero )) {
                printf( "   0.    " );
            }
            else {
                printf( " %8.4f", MAGMA_D_REAL( *A(i,j) ));
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
void magma_dprint_gpu( int m, int n, magmaDouble_ptr dA, size_t dA_offset, int ldda, magma_queue_t queue )
{
    int lda = m;
    double* A = (double*) malloc( lda*n*sizeof(double) );
    magma_dgetmatrix( m, n, dA, dA_offset, ldda,  A, 0, lda, queue );
    
    magma_dprint( m, n, A, lda );
    
    free( A );
}
