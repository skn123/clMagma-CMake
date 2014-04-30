/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from testing_zposv.cpp normal z -> d, Fri Jan 10 15:51:19 2014
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dposv
*/
int main( int argc, char** argv)
{
    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_R, *h_B, *h_X;
    magma_int_t N, lda, ldb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    /* Initialize */
    magma_queue_t  queue[2];
    magma_device_t device[ MagmaMaxGPUs ];
    int num = 0;
    magma_err_t err;
    magma_init();

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    double tol = opts.tolerance * lapackf77_dlamch("E");

    err = magma_get_devices( device, MagmaMaxGPUs, &num );
    if ( err != 0 || num < 1 ) {
      fprintf( stderr, "magma_get_devices failed: %d\n", err );
      exit(-1);
    }

    // Create two queues on device opts.device
    err = magma_queue_create( device[ opts.device ], &queue[0] );
    if ( err != 0 ) {
      fprintf( stderr, "magma_queue_create failed: %d\n", err );
      exit(-1);
    }
    err = magma_queue_create( device[ opts.device ], &queue[1] );
    if ( err != 0 ) {
      fprintf( stderr, "magma_queue_create failed: %d\n", err );
      exit(-1);
    }

    printf("ngpu %d, uplo %s\n", (int) opts.ngpu, lapack_uplo_const( opts.uplo ) );
    printf("    N  NRHS   CPU Gflop/s (sec)   GPU GFlop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[i];
            lda = ldb = N;
            gflops = ( FLOPS_DPOTRF( N ) + FLOPS_DPOTRS( N, opts.nrhs ) ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A, double, lda*N         );
            TESTING_MALLOC_PIN( h_R, double, lda*N         );
            TESTING_MALLOC_CPU( h_B, double, ldb*opts.nrhs );
            TESTING_MALLOC_CPU( h_X, double, ldb*opts.nrhs );
            TESTING_MALLOC_CPU( work, double,            N             );
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            sizeA = lda*N;
            sizeB = ldb*opts.nrhs;
            lapackf77_dlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_dlarnv( &ione, ISEED, &sizeB, h_B );
            magma_dmake_hpd( N, h_A, lda );
            
            // copy A to R and B to X; save A and B for residual
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N,         h_A, &lda, h_R, &lda );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &opts.nrhs, h_B, &ldb, h_X, &ldb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_dposv( opts.uplo, N, opts.nrhs, h_R, lda, h_X, ldb, &info, queue );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_dpotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Residual
               =================================================================== */
            Anorm = lapackf77_dlange("I", &N, &N,         h_A, &lda, work);
            Xnorm = lapackf77_dlange("I", &N, &opts.nrhs, h_X, &ldb, work);
            
            blasf77_dgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &opts.nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb );
            
            Rnorm = lapackf77_dlange("I", &N, &opts.nrhs, h_B, &ldb, work);
            error = Rnorm/(N*Anorm*Xnorm);
            status |= ! (error < tol);
            
            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_dposv( lapack_uplo_const(opts.uplo), &N, &opts.nrhs, 
                                 h_A, &lda, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_dposv returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                printf( "%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e%s\n",
                        (int) N, (int) opts.nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (error < tol ? "" : "  failed"));
            }
            else {
                printf( "%5d %5d     ---   (  ---  )   %7.2f (%7.2f)   %8.2e%s\n",
                        (int) N, (int) opts.nrhs, gpu_perf, gpu_time,
                        error, (error < tol ? "" : "  failed"));
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_X );
            TESTING_FREE_CPU( work );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_queue_destroy( queue[0] );
    magma_queue_destroy( queue[1] );
    magma_finalize();

    return status;
}
