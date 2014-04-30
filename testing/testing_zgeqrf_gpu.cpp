/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @precisions normal z -> s d c

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

// Flops formula
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6.*FMULS_GEQRF(m, n) + 2.*FADDS_GEQRF(m, n) )
#else
#define FLOPS(m, n) (    FMULS_GEQRF(m, n) +    FADDS_GEQRF(m, n) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/

int main( int argc, char** argv)
{
    
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    double           matnorm, work[1];
    magmaDoubleComplex  mzone = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R, *tau, *hwork, tmp[1];
    magmaDoubleComplex_ptr d_A;

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2, lda, ldda, lhwork;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10176};

    magma_int_t i, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
        }
        if ( M == 0 ) {
            M = N;
        }
        if ( N == 0 ) {
            N = M;
        }
        if (M>0 && N>0)
            printf("  testing_zgeqrf_gpu -M %d -N %d\n\n", M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_zgeqrf_gpu -M %d -N %d\n\n", 1024, 1024);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgeqrf_gpu -M %d -N %d\n\n", 1024, 1024);
        M = N = size[7];
    }

    /* Initialize */
    magma_queue_t  queue1, queue2;
    magma_device_t device[ MagmaMaxGPUs ];
    int num = 0;
    magma_err_t err;

    magma_init();
    err = magma_get_devices( device, MagmaMaxGPUs, &num );
    if ( err != 0 || num < 1 ) {
      fprintf( stderr, "magma_get_devices failed: %d\n", err );
      exit(-1);
    }
    err = magma_queue_create( device[0], &queue1 );
    if ( err != 0 ) {
      fprintf( stderr, "magma_queue_create failed: %d\n", err );
      exit(-1);
    }
    err = magma_queue_create( device[0], &queue2 );
    if ( err != 0 ) {
      fprintf( stderr, "magma_queue_create failed: %d\n", err );
      exit(-1);
    }

    magma_queue_t queues[2] = {queue1, queue2};

    ldda   = ((M+31)/32)*32;
    n2     = M * N;
    min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    TESTING_MALLOC_CPU( tau, magmaDoubleComplex, min_mn );
    TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, n2     );
    TESTING_MALLOC_PIN( h_R, magmaDoubleComplex, n2     );
    TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*N );

    lhwork = -1;
    lapackf77_zgeqrf(&M, &N, h_A, &M, tau, tmp, &lhwork, &info);
    lhwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );

    TESTING_MALLOC_CPU( hwork, magmaDoubleComplex, lhwork );

    printf("\n\n");
    printf("  M     N    CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
    printf("======================================================================\n");
    for(i=0; i<8; i++){
        if (argc == 1){
            M = N = size[i];
        }
        min_mn= min(M, N);
        lda   = M;
        n2    = lda*N;
        ldda  = ((M+31)/32)*32;
        gflops = FLOPS( (double)M, (double)N ) * 1e-9;

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_zgeqrf(&M, &N, h_A, &M, tau, hwork, &lhwork, &info);
        cpu_time = magma_wtime() - cpu_time;
        if (info < 0)
            printf("Argument %d of lapack_zgeqrf had an illegal value.\n", -info);

        cpu_perf = gflops / cpu_time;

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_zsetmatrix( M, N, h_R, 0, lda, d_A, 0, ldda, queue1 );
        magma_zgeqrf2_gpu( M, N, d_A, 0, ldda, tau, &info, queues);

        magma_zsetmatrix( M, N, h_R, 0, lda, d_A, 0, ldda, queue1 );
        clFinish(queue1);
        clFinish(queue2);

        gpu_time = magma_wtime();
        magma_zgeqrf2_gpu( M, N, d_A, 0, ldda, tau, &info, queues);
        gpu_time = magma_wtime() - gpu_time;

        if (info < 0)
          printf("Argument %d of magma_zgeqrf2 had an illegal value.\n", -info);
        
        gpu_perf = gflops / gpu_time;
        
        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        magma_zgetmatrix( M, N, d_A, 0, ldda, h_R, 0, M, queue1 );
        
        matnorm = lapackf77_zlange("f", &M, &N, h_A, &M, work);
        blasf77_zaxpy(&n2, &mzone, h_A, &ione, h_R, &ione);
        
        printf("%5d %5d   %6.2f (%6.2f)     %6.2f (%6.2f)       %e\n",
               M, N, cpu_perf, cpu_time, gpu_perf, gpu_time,
               lapackf77_zlange("f", &M, &N, h_R, &M, work) / matnorm);
        
        if (argc != 1)
          break;
    }
    
    /* clean up */
    TESTING_FREE_CPU( tau );
    TESTING_FREE_CPU( h_A );
    TESTING_FREE_CPU( hwork );
    TESTING_FREE_PIN( h_R );
    TESTING_FREE_DEV( d_A );

    magma_queue_destroy( queue1 );
    magma_queue_destroy( queue2 );

    magma_finalize();
}
