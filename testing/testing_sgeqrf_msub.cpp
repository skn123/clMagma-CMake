/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from testing_zgeqrf_msub.cpp normal z -> s, Fri Jan 10 15:51:20 2014

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

void init_matrix( int m, int n, float *h_A, magma_int_t lda )
{
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    lapackf77_slarnv( &ione, ISEED, &n2, h_A );
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgeqrf_msub
*/

int main( int argc, char** argv)
{
    real_Double_t gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;

    float  matnorm, work[1];
    float  c_neg_one = MAGMA_S_NEG_ONE;
    float *h_A, *h_R, *tau, *h_work, tmp[1];
    magmaFloat_ptr d_lA[MagmaMaxGPUs];

    /* Matrix size */
    magma_int_t M = 0, N = 0, flag = 0, n2, check = 0;
    magma_int_t n_local[MagmaMaxGPUs*MagmaMaxSubs], lda, ldda, lhwork;
    magma_int_t size[10] = {1000,2000,3000,4000,5000,6000,7000,8000,9000,10000};

    magma_int_t i, info, min_mn, nb;
    int num_gpus = 1, num_subs = 1, tot_subs = 1;
    magma_int_t ione = 1;

    M = N = size[9];
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                N = atoi(argv[++i]);
                flag = 1;
            } else if (strcmp("-M", argv[i])==0) {
                M = atoi(argv[++i]);
                flag = 1;
            } else if (strcmp("-NGPU", argv[i])==0) {
                num_gpus = atoi(argv[++i]);
            } else if (strcmp("-NSUB", argv[i])==0) {
                num_subs = atoi(argv[++i]);
            } else if (strcmp("-check", argv[i])==0) {
                check = 1;
            }
        }
        if ( M == 0 ) {
            M = N;
        }
        if ( N == 0 ) {
            N = M;
        }
    }
    

    if (num_gpus > MagmaMaxGPUs){
      printf("More GPUs requested than available. Have to change it.\n");
      num_gpus = MagmaMaxGPUs;
    }
    if (num_subs > MagmaMaxSubs) {
      printf("More buffers requested than available. Have to change it.\n");
      num_subs = MagmaMaxSubs;
    }
    tot_subs = num_gpus * num_subs;

    printf("\nNumber of GPUs to be used = %d\n", (int) num_gpus);
    printf("Usage: \n");
    printf("  testing_sgeqrf_msub -M %d -N %d -NGPU %d -NSUB %d %s\n\n", M, N, num_gpus, num_subs, (check == 1 ? "-check" : " "));

    /* Initialize */
    magma_queue_t  queues[2*MagmaMaxGPUs];
    magma_device_t devices[MagmaMaxGPUs];
    int num = 0;
    magma_err_t err;
    magma_init();
    err = magma_get_devices( devices, MagmaMaxGPUs, &num );
    if ( err != 0 || num < 1 ) {
        fprintf( stderr, "magma_get_devices failed: %d\n", err );
        exit(-1);
    }
    for (i=0; i<num_gpus; i++){
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
    printf( "\n" );
    
    printf("  M     N     CPU GFlop/s (sec.)     GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
    printf("==========================================================================\n");
    for(i=0; i<10; i++){
        if (flag == 0) {
            M = N = size[i];
        }
        nb     = magma_get_sgeqrf_nb(M);
        min_mn = min(M, N);
        lda    = M;
        n2     = lda*N;
        ldda   = ((M+31)/32)*32;
        gflops = FLOPS_SGEQRF( (float)M, (float)N ) / 1e9;

        /* Allocate host memory for the matrix */
        TESTING_MALLOC_CPU( tau, float, min_mn );
        TESTING_MALLOC_CPU( h_R, float, n2 );

        /* Allocate host workspace */
        lhwork = -1;
        lapackf77_sgeqrf(&M, &N, h_R, &M, tau, tmp, &lhwork, &info);
        lhwork = (magma_int_t)MAGMA_S_REAL( tmp[0] );
        TESTING_MALLOC_CPU( h_work, float, lhwork );

        /* Allocate device memory for the matrix */
        for (int j=0; j<tot_subs; j++) {      
            n_local[j] = ((N/nb)/tot_subs)*nb;
            if (j < (N/nb)%tot_subs)
                n_local[j] += nb;
            else if (j == (N/nb)%tot_subs)
                n_local[j] += N%nb;
      
            TESTING_MALLOC_DEV( d_lA[j], float, ldda*n_local[j] );
        }

        /* Initialize the matrix */
        init_matrix( M, N, h_R, lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_queue_t *trans_queues = (magma_queue_t*)malloc(tot_subs*sizeof(magma_queue_t));
        for (int j=0; j<tot_subs; j++) {
            trans_queues[j] = queues[2*(j%num_gpus)];
        }
        
        // warm-up
        magmablas_ssetmatrix_1D_bcyclic(M, N, h_R, lda, d_lA, ldda, tot_subs, nb, trans_queues);
        magma_sgeqrf_msub(num_subs, num_gpus, M, N, d_lA, ldda, tau, &info, queues);

        magmablas_ssetmatrix_1D_bcyclic(M, N, h_R, lda, d_lA, ldda, tot_subs, nb, trans_queues);
        gpu_time = magma_wtime();
        magma_sgeqrf_msub(num_subs, num_gpus, M, N, d_lA, ldda, tau, &info, queues);
        gpu_time = magma_wtime() - gpu_time;
        gpu_perf = gflops / gpu_time;

        if (info < 0)
          printf("Argument %d of magma_sgeqrf_msub had an illegal value.\n", (int) -info);
        
        if (check == 1) {
            /* =====================================================================
               Check the result compared to LAPACK
               =================================================================== */
            magmablas_sgetmatrix_1D_bcyclic(M, N, d_lA, ldda, h_R, lda, tot_subs, nb, trans_queues);
            TESTING_MALLOC_CPU( h_A, float, n2 );
            init_matrix( M, N, h_A, lda );

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_sgeqrf(&M, &N, h_A, &lda, tau, h_work, &lhwork, &info);
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;

            if (info < 0)
                printf("Argument %d of lapack_sgeqrf had an illegal value.\n", (int) -info);

            matnorm = lapackf77_slange("f", &M, &N, h_A, &M, work);
            blasf77_saxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
        
            printf("%5d %5d      %6.2f (%6.2f)       %6.2f (%6.2f)       %e\n",
                   (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                   lapackf77_slange("f", &M, &N, h_R, &M, work) / matnorm);

            TESTING_FREE_PIN( h_A );
        } else {
            printf("%5d %5d            -- ( -- )       %6.2f (%6.2f)           --\n",
                   (int) M, (int) N, gpu_perf, gpu_time );
        }
        /* Memory clean up */
        TESTING_FREE_PIN( tau );
        TESTING_FREE_PIN( h_work );
        TESTING_FREE_PIN( h_R );
        for (int j=0; j<tot_subs; j++) {
            TESTING_FREE_DEV( d_lA[j] );
        }

        if (flag != 0)
          break;
    }
    
    for (i=0; i<num_gpus; i++) {
        magma_queue_destroy(queues[2*i]);
        magma_queue_destroy(queues[2*i+1]);
    }

    /* Shutdown */
    magma_finalize();
}
