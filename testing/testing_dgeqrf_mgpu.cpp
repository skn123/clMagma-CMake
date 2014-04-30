/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from testing_zgeqrf_mgpu.cpp normal z -> d, Fri Jan 10 15:51:20 2014

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
#define PRECISION_d
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6.*FMULS_GEQRF(m, n) + 2.*FADDS_GEQRF(m, n) )
#else
#define FLOPS(m, n) (    FMULS_GEQRF(m, n) +    FADDS_GEQRF(m, n) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgeqrf_mgpu
*/

int main( int argc, char** argv)
{
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time, error;

    double           matnorm, work[1];
    double  c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_R, *tau, *h_work, tmp[1];
    magmaDouble_ptr d_lA[MagmaMaxGPUs];

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2, n_local[4], lda, ldda, lhwork;
    magma_int_t size[10] = {1000,2000,3000,4000,5000,6000,7000,8000,9000,10000};

    magma_int_t i, k, nk, info, min_mn;
    int max_num_gpus = 2, num_gpus = 2;
    
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
            else if (strcmp("-NGPU", argv[i])==0)
              num_gpus = atoi(argv[++i]);
        }
        if ( M == 0 ) {
            M = N;
        }
        if ( N == 0 ) {
            N = M;
        }
        if (M>0 && N>0)
          printf("  testing_dgeqrf_gpu -M %d -N %d -NGPU %d\n\n", (int) M, (int) N, (int) num_gpus);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_dgeqrf_gpu -M %d -N %d -NGPU %d\n\n", 
                       1024, 1024, 1);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_dgeqrf_gpu -M %d -N %d -NGPU %d\n\n", 1024, 1024, 1);
        M = N = size[9];
    }
    
    ldda   = ((M+31)/32)*32;
    n2     = M * N;
    min_mn = min(M, N);

    magma_int_t nb  = magma_get_dgeqrf_nb(M);

    if (num_gpus > max_num_gpus){
      printf("More GPUs requested than available. Have to change it.\n");
      num_gpus = max_num_gpus;
    }
    printf("Number of GPUs to be used = %d\n", (int) num_gpus);

    /* Initialize */
    magma_queue_t  queues[MagmaMaxGPUs * 2];
    magma_device_t devices[ MagmaMaxGPUs ];
    int num = 0;
    magma_err_t err;
    magma_init();
    err = magma_get_devices( devices, MagmaMaxGPUs, &num );
    if ( err != 0 || num < 1 ) {
        fprintf( stderr, "magma_get_devices failed: %d\n", err );
        exit(-1);
    }
    for(i=0;i<num_gpus;i++){
        err = magma_queue_create( devices[i], &queues[2*i] );
        if ( err != 0 ) {
            fprintf( stderr, "magma_queue_create failed: %d\n", err );
            exit(-1);
        }
        err = magma_queue_create( devices[i], &queues[2*i+1] );
        if ( err != 0 ) {
            fprintf( stderr, "magma_queue_create failed: %d\n", err );
            exit(-1);
        }
    }
    
    /* Allocate host memory for the matrix */
    TESTING_MALLOC_CPU( tau, double, min_mn );
    TESTING_MALLOC_CPU( h_A, double, n2     );
    TESTING_MALLOC_CPU( h_R, double, n2     );

    for(i=0; i<num_gpus; i++){      
        n_local[i] = ((N/nb)/num_gpus)*nb;
        if (i < (N/nb)%num_gpus)
            n_local[i] += nb;
        else if (i == (N/nb)%num_gpus)
            n_local[i] += N%nb;
        
        TESTING_MALLOC_DEV( d_lA[i], double, ldda*n_local[i] );
        printf("device %2d n_local = %4d\n", (int) i, (int) n_local[i]);  
    }

    lhwork = -1;
    lapackf77_dgeqrf(&M, &N, h_A, &M, tau, tmp, &lhwork, &info);
    lhwork = (magma_int_t)MAGMA_D_REAL( tmp[0] );

    TESTING_MALLOC_CPU( h_work, double, lhwork );

    printf("  M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
    printf("======================================================================\n");
    for(i=0; i<10; i++){
        if (argc == 1){
            M = N = size[i];
        }
        min_mn= min(M, N);
        lda   = M;
        n2    = lda*N;
        ldda  = ((M+31)/32)*32;
        gflops = FLOPS( (double)M, (double)N ) * 1e-9;

        /* Initialize the matrix */
        lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_dgeqrf(&M, &N, h_A, &M, tau, h_work, &lhwork, &info);
        cpu_time = magma_wtime() - cpu_time;
        if (info < 0)
            printf("Argument %d of lapack_dgeqrf had an illegal value.\n", (int) -info);

        cpu_perf = gflops / cpu_time;

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        int j;
        magma_queue_t *trans_queues = (magma_queue_t*)malloc(num_gpus*sizeof(magma_queue_t));
        for(j=0;j<num_gpus;j++){
            trans_queues[j] = queues[2*j];
        }
        
        // warm-up
        magmablas_dsetmatrix_1D_bcyclic(M, N, h_R, lda, d_lA, ldda, num_gpus, nb, trans_queues);
        magma_dgeqrf2_mgpu( num_gpus, M, N, d_lA, ldda, tau, &info, queues);

        magmablas_dsetmatrix_1D_bcyclic(M, N, h_R, lda, d_lA, ldda, num_gpus, nb, trans_queues);
        gpu_time = magma_wtime();
        magma_dgeqrf2_mgpu( num_gpus, M, N, d_lA, ldda, tau, &info, queues);
        gpu_time = magma_wtime() - gpu_time;

        if (info < 0)
          printf("Argument %d of magma_dgeqrf2 had an illegal value.\n", (int) -info);
        
        gpu_perf = gflops / gpu_time;
        
        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        magmablas_dgetmatrix_1D_bcyclic(M, N, d_lA, ldda, h_R, lda, num_gpus, nb, trans_queues);
        
        matnorm = lapackf77_dlange("f", &M, &N, h_A, &M, work);
        blasf77_daxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
        
        printf("%5d %5d  %6.2f (%6.2f)        %6.2f (%6.2f)       %e\n",
               (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
               lapackf77_dlange("f", &M, &N, h_R, &M, work) / matnorm);
        
        if (argc != 1)
          break;
    }
    
    /* Memory clean up */
    TESTING_FREE_PIN( tau );
    TESTING_FREE_PIN( h_A );
    TESTING_FREE_PIN( h_work );
    TESTING_FREE_PIN( h_R );

    for(i=0; i<num_gpus; i++){
        TESTING_FREE_DEV( d_lA[i] );
        magma_queue_destroy(queues[2*i]);
        magma_queue_destroy(queues[2*i+1]);
    }

    /* Shutdown */
    magma_finalize();
}
