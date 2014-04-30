/*
 *  -- clMAGMA (version 1.1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date January 2014
 *
 * @generated from testing_zpotrf_mgpu.cpp normal z -> d, Fri Jan 10 15:51:19 2014
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

#define PRECISION_d
// Flops formula
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_POTRF(n) + 2. * FADDS_POTRF(n) )
#else
#define FLOPS(n) (      FMULS_POTRF(n) +      FADDS_POTRF(n) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dpotrf_mgpu
*/
#define h_A(i,j) h_A[ i + j*lda ]

int main( int argc, char** argv)
{
    real_Double_t gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    double *h_A, *h_R;
    magmaDouble_ptr d_lA[MagmaMaxGPUs];
    magma_int_t N = 0, n2, lda, ldda;
    magma_int_t size[10] =
        { 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };
    
    magma_int_t i, j, k, info;
    double mz_one = MAGMA_D_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      work[1], matnorm, diffnorm;
   
    magma_int_t num_gpus0 = 1, num_gpus, flag = 0;
    int nb, mb, n_local, nk;

    magma_uplo_t uplo = MagmaLower;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0){
                N = atoi(argv[++i]);
                if (N>0) {
                    size[0] = size[9] = N;
                    flag = 1;
                }else exit(1);
            }
            if(strcmp("-NGPU", argv[i])==0)
                num_gpus0 = atoi(argv[++i]);
            if(strcmp("-UPLO", argv[i])==0){
                if(strcmp("L", argv[++i])==0){
                    uplo = MagmaLower;
                }else{
                    uplo = MagmaUpper;
                }            
            }
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_dpotrf_mgpu -N %d -NGPU %d -UPLO -L\n\n", 1024, num_gpus0);
    }

    /* looking for max. ldda */
    ldda = 0;
    n2 = 0;
    for(i=0;i<10;i++){
        N = size[i];
        nb = magma_get_dpotrf_nb(N);
        mb = nb;
        if(num_gpus0 > N/nb){
            num_gpus = N/nb;
            if(N%nb != 0) num_gpus ++;
        }else{
            num_gpus = num_gpus0;
        }
        n_local = nb*(1+N/(nb*num_gpus))*mb*((N+mb-1)/mb);
        if(n_local > ldda) ldda = n_local;
        if(n2 < N*N) n2 = N*N;
        if(flag != 0) break;
    }

     /* Allocate host memory for the matrix */
    TESTING_MALLOC_PIN( h_A, double, n2 );
    TESTING_MALLOC_PIN( h_R, double, n2 );

    /* Initialize */
    magma_queue_t  queues[MagmaMaxGPUs * 2];
    //magma_queue_t  queues[MagmaMaxGPUs];
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

    printf("each buffer size: %d\n", ldda);
    /* allocate local matrix on Buffers */
    for(i=0; i<num_gpus0; i++){
        TESTING_MALLOC_DEV( d_lA[i], double, ldda );
    }

    
    printf("\n\n");
    printf("Using GPUs: %d\n", num_gpus0);
    if(uplo == MagmaUpper){
        printf("\n  testing_dpotrf_mgpu -N %d -NGPU %d -UPLO U\n\n", N, num_gpus0);
    }else{
        printf("\n  testing_dpotrf_mgpu -N %d -NGPU %d -UPLO L\n\n", N, num_gpus0);
    }
            printf("  N    CPU GFlop/s (sec)    GPU GFlop/s (sec)    ||R_magma-R_lapack||_F / ||R_lapack||_F\n");
    printf("========================================================================================\n");
    for(i=0; i<10; i++){
        N   = size[i];
        lda = N;
        n2  = lda*N;
        ldda = ((N+31)/32)*32;
        gflops = FLOPS( (double)N ) * 1e-9;
        
        /* Initialize the matrix */
        lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
        /* Symmetrize and increase the diagonal */
        for( int i = 0; i < N; ++i ) {
            MAGMA_D_SET2REAL( h_A(i,i), MAGMA_D_REAL(h_A(i,i)) + N );
            for( int j = 0; j < i; ++j ) {
          h_A(i, j) = MAGMA_D_CNJG( h_A(j,i) );
            }
        }
        lapackf77_dlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );

        /* Warm up to measure the performance */
        nb = magma_get_dpotrf_nb(N);
        if(num_gpus0 > N/nb){
            num_gpus = N/nb;
            if(N%nb != 0) num_gpus ++;
            printf("too many GPUs for the matrix size, using %d GPUs\n", (int)num_gpus);
        }else{
            num_gpus = num_gpus0;
        }
        /* distribute matrix to gpus */
        if(uplo == MagmaUpper){
            // Upper
            ldda = ((N+mb-1)/mb)*mb;    
            for(j=0;j<N;j+=nb){
                k = (j/nb)%num_gpus;
                nk = min(nb, N-j);
                magma_dsetmatrix(N, nk, 
                                 &h_A[j*lda], 0, lda,
                                 d_lA[k], j/(nb*num_gpus)*nb*ldda, ldda, 
                                 queues[2*k]);
            }
        }else{
            // Lower
            ldda = (1+N/(nb*num_gpus))*nb;
            for(j=0;j<N;j+=nb){
                k = (j/nb)%num_gpus;
                nk = min(nb, N-j);
                magma_dsetmatrix(nk, N, &h_A[j], 0, lda,
                                    d_lA[k], (j/(nb*num_gpus)*nb), ldda,
                                    queues[2*k]);
            }
        }

        magma_dpotrf_mgpu( num_gpus, uplo, N, d_lA, 0, ldda, &info, queues );
        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        /* distribute matrix to gpus */
        if(uplo == MagmaUpper){
            // Upper
            ldda = ((N+mb-1)/mb)*mb;    
            for(j=0;j<N;j+=nb){
                k = (j/nb)%num_gpus;
                nk = min(nb, N-j);
                magma_dsetmatrix(N, nk, 
                                 &h_A[j*lda], 0, lda,
                                 d_lA[k], j/(nb*num_gpus)*nb*ldda, ldda, 
                                 queues[2*k]);
            }
        }else{
            // Lower
            ldda = (1+N/(nb*num_gpus))*nb;
            for(j=0;j<N;j+=nb){
                k = (j/nb)%num_gpus;
                nk = min(nb, N-j);
                magma_dsetmatrix(nk, N, &h_A[j], 0, lda,
                                    d_lA[k], (j/(nb*num_gpus)*nb), ldda,
                                    queues[2*k]);
            }
        }
    
        gpu_time = magma_wtime();
        magma_dpotrf_mgpu( num_gpus, uplo, N, d_lA, 0, ldda, &info, queues );
        gpu_time = magma_wtime() - gpu_time;
        if (info != 0)
            printf( "magma_dpotrf had error %d.\n", info );

        gpu_perf = gflops / gpu_time;
       
        /* gather matrix from gpus */
        if(uplo==MagmaUpper){
            // Upper
            for(j=0;j<N;j+=nb){
                k = (j/nb)%num_gpus;
                nk = min(nb, N-j);
                magma_dgetmatrix(N, nk,
                                 d_lA[k], j/(nb*num_gpus)*nb*ldda, ldda,
                                 &h_R[j*lda], 0, lda, queues[2*k]);
            }
        }else{
            // Lower
            for(j=0; j<N; j+=nb){
                k = (j/nb)%num_gpus;
                nk = min(nb, N-j);
                magma_dgetmatrix( nk, N, 
                            d_lA[k], (j/(nb*num_gpus)*nb), ldda, 
                            &h_R[j], 0, lda, queues[2*k] );
            }
        }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        if(uplo == MagmaLower){
            lapackf77_dpotrf( MagmaLowerStr, &N, h_A, &lda, &info );
        }else{
            lapackf77_dpotrf( MagmaUpperStr, &N, h_A, &lda, &info );
        }
        cpu_time = magma_wtime() - cpu_time;
        if (info != 0)
            printf( "lapackf77_dpotrf had error %d.\n", info );
        
        cpu_perf = gflops / cpu_time;
        /* =====================================================================
           Check the result compared to LAPACK
           |R_magma - R_lapack| / |R_lapack|
           =================================================================== */
        matnorm = lapackf77_dlange("f", &N, &N, h_A, &lda, work);
        blasf77_daxpy(&n2, &mz_one, h_A, &ione, h_R, &ione);
        diffnorm = lapackf77_dlange("f", &N, &N, h_R, &lda, work);
        printf( "%5d     %6.2f (%6.2f)     %6.2f (%6.2f)         %e\n",
                N, cpu_perf, cpu_time, gpu_perf, gpu_time, diffnorm / matnorm );
        
        if (flag != 0)
            break;
    }

    /* clean up */
    TESTING_FREE_PIN( h_A );
    TESTING_FREE_PIN( h_R );
    for(i=0;i<num_gpus;i++){
        TESTING_FREE_DEV( d_lA[i] );
        magma_queue_destroy( queues[2*i]   );
        magma_queue_destroy( queues[2*i+1] );
    }
    magma_finalize();
}
