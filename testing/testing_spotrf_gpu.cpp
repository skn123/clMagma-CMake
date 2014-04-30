/*
 *  -- clMAGMA (version 1.1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date January 2014
 *
 * @generated from testing_zpotrf_gpu.cpp normal z -> s, Fri Jan 10 15:51:19 2014
 *
 **/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_s
// Flops formula
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_POTRF(n) + 2. * FADDS_POTRF(n) )
#else
#define FLOPS(n) (      FMULS_POTRF(n) +      FADDS_POTRF(n) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing spotrf
*/
#define hA(i,j) hA[ i + j*lda ]

int main( int argc, char** argv)
{
    real_Double_t gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    float *hA, *hR;
    magmaFloat_ptr dA;
    magma_int_t N = 0, n2, lda, ldda;
    magma_int_t size[10] =
        { 1024, 2048, 3072, 4032, 5184, 6048, 7200, 8064, 8928, 10560 };
    
    magma_int_t i, info;
    float mz_one = MAGMA_S_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    float      work[1], matnorm, diffnorm;
    
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
        }
        if (N>0) size[0] = size[9] = N;
        else exit(1);
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_spotrf_gpu -N %d\n\n", 1024);
    }

    /* Initialize */
    magma_queue_t  queue;
    magma_device_t device[ MagmaMaxGPUs ];
    int num = 0;
    magma_err_t err;
    magma_init();
    err = magma_get_devices( device, MagmaMaxGPUs, &num );
    if ( err != 0 || num < 1 ) {
        fprintf( stderr, "magma_get_devices failed: %d\n", err );
        exit(-1);
    }
    err = magma_queue_create( device[0], &queue );
    if ( err != 0 ) {
        fprintf( stderr, "magma_queue_create failed: %d\n", err );
        exit(-1);
    }

    /* Allocate memory for the largest matrix */
    N    = size[9];
    n2   = N * N;
    ldda = ((N+31)/32) * 32;
    TESTING_MALLOC_CPU( hA, float, n2 );
    TESTING_MALLOC_PIN( hR, float, n2 );
    TESTING_MALLOC_DEV( dA, float, ldda*N );
    
    printf("\n\n");
    printf("  N    CPU GFlop/s (sec)    GPU GFlop/s (sec)    ||R_magma-R_lapack||_F / ||R_lapack||_F\n");
    printf("========================================================================================\n");
    for(i=0; i<10; i++){
        N   = size[i];
        lda = N;
        n2  = lda*N;
        ldda = ((N+31)/32)*32;
        gflops = FLOPS( (float)N ) * 1e-9;
        
        /* Initialize the matrix */
        lapackf77_slarnv( &ione, ISEED, &n2, hA );
        /* Symmetrize and increase the diagonal */
        for( int i = 0; i < N; ++i ) {
            MAGMA_S_SET2REAL( hA(i,i), MAGMA_S_REAL(hA(i,i)) + N );
            for( int j = 0; j < i; ++j ) {
          hA(i, j) = MAGMA_S_CNJG( hA(j,i) );
            }
        }
        lapackf77_slacpy( MagmaFullStr, &N, &N, hA, &lda, hR, &lda );

        /* Warm up to measure the performance */
        magma_ssetmatrix( N, N, hA, 0, lda, dA, 0, ldda, queue );
        magma_spotrf_gpu( MagmaUpper, N, dA, 0, ldda, &info, queue );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_ssetmatrix( N, N, hA, 0, lda, dA, 0, ldda, queue );
        gpu_time = magma_wtime();
        magma_spotrf_gpu( MagmaUpper, N, dA, 0, ldda, &info, queue );
        gpu_time = magma_wtime() - gpu_time;
        if (info != 0)
            printf( "magma_spotrf had error %d.\n", info );

        gpu_perf = gflops / gpu_time;
        
        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_spotrf( MagmaUpperStr, &N, hA, &lda, &info );
        cpu_time = magma_wtime() - cpu_time;
        if (info != 0)
            printf( "lapackf77_spotrf had error %d.\n", info );
        
        cpu_perf = gflops / cpu_time;
        
        /* =====================================================================
           Check the result compared to LAPACK
           |R_magma - R_lapack| / |R_lapack|
           =================================================================== */
        magma_sgetmatrix( N, N, dA, 0, ldda, hR, 0, lda, queue );
        matnorm = lapackf77_slange("f", &N, &N, hA, &lda, work);
        blasf77_saxpy(&n2, &mz_one, hA, &ione, hR, &ione);
        diffnorm = lapackf77_slange("f", &N, &N, hR, &lda, work);
        printf( "%5d     %6.2f (%6.2f)     %6.2f (%6.2f)         %e\n",
                N, cpu_perf, cpu_time, gpu_perf, gpu_time, diffnorm / matnorm );
        
        if (argc != 1)
            break;
    }

    /* clean up */
    TESTING_FREE_CPU( hA );
    TESTING_FREE_PIN( hR );
    TESTING_FREE_DEV( dA );
    magma_queue_destroy( queue );
    magma_finalize();
}
