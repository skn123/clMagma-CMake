/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from testing_zgetri_gpu.cpp normal z -> s, Fri Jan 10 15:51:19 2014

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
   -- Testing sgetrf
*/
int main( int argc, char** argv)
{
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time, error;
    float *h_A, *h_R;
    magmaFloat_ptr d_A, dwork;
    magma_int_t N = 0, n2, lda, ldda;
    magma_int_t size[10] = { 1024, 2048, 3072, 4032, 5184, 5600, 5600, 5600, 5600, 5600 };
    magma_int_t ntest = 10;
    
    magma_int_t i, info;
    float c_neg_one = MAGMA_S_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0, 0, 0, 1};
    float *work;
    float tmp;
    float rwork[1];
    magma_int_t *ipiv;
    magma_int_t lwork, ldwork;
    float A_norm, R_norm;
    
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
        }
        if (N>0) size[0] = size[ntest-1] = N;
        else exit(1);
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_sgetri_gpu -N %d\n\n", 1024);
    }
    
    /* query for Lapack workspace size */
    N     = size[ntest-1];
    lda   = N;
    work  = &tmp;
    lwork = -1;
    lapackf77_sgetri( &N, h_A, &lda, ipiv, work, &lwork, &info );
    if (info != 0)
        printf("lapackf77_sgetri returned error %d\n", (int) info);
    lwork = int( MAGMA_S_REAL( *work ));

    /* query for Magma workspace size */
    ldwork = N * magma_get_sgetri_nb( N );

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
    
    /* Allocate memory */
    n2   = N * N;
    ldda = ((N+31)/32) * 32;
    TESTING_MALLOC_CPU( ipiv,  magma_int_t,        N      );
    TESTING_MALLOC_CPU( work,  float, lwork  );
    TESTING_MALLOC_CPU( h_A,   float, n2     );
    TESTING_MALLOC_PIN( h_R,   float, n2     );
    TESTING_MALLOC_DEV( d_A,   float, ldda*N );
    TESTING_MALLOC_DEV( dwork, float, ldwork );

    printf("  N    CPU GFlop/s    GPU GFlop/s    ||R||_F / ||A||_F\n");
    printf("========================================================\n");
    for( i=0; i < ntest; i++ ){
        N   = size[i];
        lda = N;
        n2  = lda*N;
        gflops = FLOPS_SGETRI( (float)N ) / 1e9;
        
        ldda = ((N+31)/32)*32;

        /* Initialize the matrix */
        lapackf77_slarnv( &ione, ISEED, &n2, h_A );
        A_norm = lapackf77_slange( "f", &N, &N, h_A, &lda, rwork );

        /* Factor the matrix. Both MAGMA and LAPACK will use this factor. */
        magma_ssetmatrix( N, N, h_A, 0, lda, d_A, 0, ldda, queue );
        magma_sgetrf_gpu( N, N, d_A, 0, ldda, ipiv, &info, queue );
        magma_sgetmatrix( N, N, d_A, 0, ldda, h_A, 0, lda, queue );
        
        // check for exact singularity
        //h_A[ 10 + 10*lda ] = MAGMA_S_MAKE( 0.0, 0.0 );
        //magma_ssetmatrix( N, N, h_A, lda, d_A, ldda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        //warm-up
        magma_sgetri_gpu( N,    d_A, 0, ldda, ipiv, dwork, 0, ldwork, &info, queue );
        
        magma_ssetmatrix( N, N, h_A, 0, lda, d_A, 0, ldda, queue );
        gpu_time = magma_wtime();
        magma_sgetri_gpu( N,    d_A, 0, ldda, ipiv, dwork, 0, ldwork, &info, queue );
        gpu_time = magma_wtime()-gpu_time;
        if (info != 0)
            printf("magma_sgetri_gpu returned error %d\n", (int) info);

        gpu_perf = gflops / gpu_time;
        
        magma_sgetmatrix( N, N, d_A, 0, ldda, h_R, 0, lda, queue );
         
        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_sgetri( &N,     h_A, &lda, ipiv, work, &lwork, &info );
        cpu_time = magma_wtime() - cpu_time;
        if (info != 0)
            printf("lapackf77_sgetri returned error %d\n", (int) info);
        
        cpu_perf = gflops / cpu_time;
        
        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        blasf77_saxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
        R_norm = lapackf77_slange( "f", &N, &N, h_R, &lda, rwork );
        
        printf( "%5d    %6.2f         %6.2f        %e\n",
                (int) N, cpu_perf, gpu_perf, R_norm / A_norm );
        
        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE_CPU( ipiv );
    TESTING_FREE_CPU( work );
    TESTING_FREE_CPU( h_A  );
    TESTING_FREE_PIN( h_R );
    TESTING_FREE_DEV( d_A   );
    TESTING_FREE_DEV( dwork );

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
}
