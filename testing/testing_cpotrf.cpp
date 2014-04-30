/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from testing_zpotrf.cpp normal z -> c, Fri Jan 10 15:51:19 2014
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
int main( int argc, char** argv)
{
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaFloatComplex *h_A, *h_R;
    magma_int_t N, n2, lda, info;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    float      work[1], error;
    magma_int_t status = 0;

    /* Initialize */
    magma_queue_t  queue[2];
    magma_device_t devices[MagmaMaxGPUs];
    int num = 0;
    magma_err_t err;
    magma_init();

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)

    float tol = opts.tolerance * lapackf77_slamch("E");

    err = magma_get_devices( devices, MagmaMaxGPUs, &num );
    if ( err != 0 || num < 1 ) {
      fprintf( stderr, "magma_get_devices failed: %d\n", err );
      exit(-1);
    }

    // Create two queues on device opts.device
    err = magma_queue_create( devices[opts.device], &queue[0] );
    if ( err != 0 ) {
      fprintf( stderr, "magma_queue_create failed: %d\n", err );
      exit(-1);
    }
    err = magma_queue_create( devices[opts.device], &queue[1] );
    if ( err != 0 ) {
      fprintf( stderr, "magma_queue_create failed: %d\n", err );
      exit(-1);
    }

    printf("ngpu %d, uplo %s\n", (int) opts.ngpu, lapack_uplo_const(opts.uplo) );
    printf("    N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("========================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N     = opts.nsize[i];
            lda   = N;
            n2    = lda*N;
            gflops = FLOPS_CPOTRF( N ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A, magmaFloatComplex, n2 );
            TESTING_MALLOC_CPU( h_R, magmaFloatComplex, n2 );
            
            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            magma_cmake_hpd( N, h_A, lda );
            lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_cpotrf( opts.uplo, N, h_R, lda, &info, queue );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_cpotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_cpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_cpotrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                error = lapackf77_clange("f", &N, &N, h_A, &lda, work);
                blasf77_caxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                error = lapackf77_clange("f", &N, &N, h_R, &lda, work) / error;
                
                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e%s\n",
                       (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (error < tol ? "" : "  failed") );
                status |= ! (error < tol);
            }
            else {
                printf("%5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (int) N, gpu_perf, gpu_time );
            }
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_R );
        }
    }

    magma_queue_destroy( queue[0] );
    magma_queue_destroy( queue[1] );
    magma_finalize();

    return status;
}
