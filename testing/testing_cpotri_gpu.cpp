/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from testing_zpotri_gpu.cpp normal z -> c, Fri Jan 10 15:51:19 2014

*/

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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cpotrf
*/
#define h_A(i,j) h_A[ i + j*lda ]

int main( int argc, char** argv)
{
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;

    magmaFloatComplex *h_A, *h_R;
    magmaFloatComplex_ptr d_A;
    magma_int_t N = 0, n2, lda, ldda;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,5600,5600,5600,5600,5600};
    
    magma_int_t i, info;
    magma_uplo_t  uplo = MagmaUpper;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    float      work[1], matnorm;
    
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
        printf("  testing_cpotri_gpu -N %d\n\n", 1024);
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

    /* Allocate host memory for the matrix */
    n2   = size[9] * size[9];
    ldda = ((size[9]+31)/32) * 32;
    TESTING_MALLOC_CPU( h_A, magmaFloatComplex, n2 );
    TESTING_MALLOC_PIN( h_R, magmaFloatComplex, n2 );
    TESTING_MALLOC_DEV( d_A, magmaFloatComplex, ldda*size[9] );

    printf("  N    CPU GFlop/s    GPU GFlop/s    ||R||_F / ||A||_F\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){
        N   = size[i];
        lda = N;
        n2  = lda*N;
        gflops = FLOPS_CPOTRI( (float)N ) / 1e9;
        
        ldda = ((N+31)/32)*32;

        /* Initialize the matrix */
        lapackf77_clarnv( &ione, ISEED, &n2, h_A );
        /* Symmetrize and increase the diagonal */
        {
            magma_int_t i, j;
            for(i=0; i<N; i++) {
                MAGMA_C_SET2REAL( h_A[i*lda+i], ( MAGMA_C_REAL(h_A[i*lda+i]) + 1.*N ) );
                for(j=0; j<i; j++)
                    h_A(i, j) = MAGMA_C_CNJG( h_A(j,i) );
            }
        }
        lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        /* factorize matrix */
        magma_csetmatrix( N, N, h_A, 0, lda, d_A, 0, ldda, queue );
        magma_cpotrf_gpu(uplo, N, d_A, 0, ldda, &info, queue);
        magma_cgetmatrix(N, N, d_A, 0, ldda, h_A, 0, lda, queue);

        // check for exact singularity
        //magma_cgetmatrix( N, N, d_A, ldda, h_R, lda );
        //h_R[ 10 + 10*lda ] = MAGMA_C_MAKE( 0.0, 0.0 );
        //magma_csetmatrix( N, N, h_R, lda, d_A, ldda );
       
        //warm-up
     //   magma_cpotri_gpu(uplo, N, d_A, 0, ldda, &info, queue);
        
    //    magma_csetmatrix( N, N, h_A, 0, lda, d_A, 0, ldda, queue );
        gpu_time = magma_wtime();
        magma_cpotri_gpu(uplo, N, d_A, 0, ldda, &info, queue);
        gpu_time = magma_wtime()-gpu_time;
        if (info != 0)
            printf("magma_cpotri_gpu returned error %d\n", (int) info);

        gpu_perf = gflops / gpu_time;
        
        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_cpotri(lapack_const(uplo), &N, h_A, &lda, &info);
        cpu_time = magma_wtime() - cpu_time;
        if (info != 0)
            printf("lapackf77_cpotri returned error %d\n", (int) info);
        
        cpu_perf = gflops / cpu_time;
      
        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        magma_cgetmatrix( N, N, d_A, 0, ldda, h_R, 0, lda, queue );
        matnorm = lapackf77_clange("f", &N, &N, h_A, &lda, work);
        blasf77_caxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
        printf("%5d    %6.2f         %6.2f        %e\n",
               (int) size[i], cpu_perf, gpu_perf,
               lapackf77_clange("f", &N, &N, h_R, &lda, work) / matnorm);
        
        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE_CPU( h_A );
    TESTING_FREE_PIN( h_R );
    TESTING_FREE_DEV( d_A );

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
}
