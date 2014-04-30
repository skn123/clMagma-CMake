/*
 *  -- clMAGMA (version 1.1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date January 2014
 *
 * @generated from testing_zgetrf_mgpu.cpp normal z -> c, Fri Jan 10 15:51:20 2014
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

// Flops formula
#define PRECISION_c
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6. * FMULS_GETRF(m, n) + 2. * FADDS_GETRF(m, n) )
#else
#define FLOPS(m, n) (      FMULS_GETRF(m, n) +      FADDS_GETRF(m, n) )
#endif

float get_LU_error(magma_int_t M, magma_int_t N,
                    magmaFloatComplex *A,  magma_int_t lda,
                    magmaFloatComplex *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    magmaFloatComplex alpha = MAGMA_C_ONE;
    magmaFloatComplex beta  = MAGMA_C_ZERO;
    magmaFloatComplex *L, *U;
    float work[1], matnorm, residual;
                       
    TESTING_MALLOC_CPU( L, magmaFloatComplex, M*min_mn );
    TESTING_MALLOC_CPU( U, magmaFloatComplex, min_mn*N );
    memset( L, 0, M*min_mn*sizeof(magmaFloatComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaFloatComplex) );

    lapackf77_claswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_clacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_clacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for(j=0; j<min_mn; j++)
        L[j+j*M] = MAGMA_C_MAKE( 1., 0. );
    
    matnorm = lapackf77_clange("f", &M, &N, A, &lda, work);

    blasf77_cgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_C_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_clange("f", &M, &N, LU, &lda, work);

    TESTING_FREE_CPU( L );
    TESTING_FREE_CPU( U );

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgetrf_mgpu
*/
int main( int argc, char** argv)
{
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time, error;
    magmaFloatComplex *h_A, *h_R;
    magmaFloatComplex_ptr d_lA[MagmaMaxGPUs];
    magma_int_t     *ipiv;

    /* Matrix size */
    magma_int_t M = 0, N = 0, flag = 0, n2, lda, ldda, num_gpus, num_gpus0 = 1, n_local;
    magma_int_t size[10] = {1000,2000,3000,4000,5000,6000,7000,8000,8160,8192};
    magma_int_t n_size = 10;

    magma_int_t i, k, info, min_mn, nb0, nb, nk, maxn, ret, ldn_local;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                    N = atoi(argv[++i]);
                    flag = 1;
            } else if (strcmp("-M", argv[i])==0) {
                    M = atoi(argv[++i]);
                    flag = 1;
            } else if (strcmp("-NGPU", argv[i])==0)
                num_gpus0 = atoi(argv[++i]);
        }
    }
    if( flag != 0 ) {
        if (M>0 && N>0 && num_gpus0>0)
            printf("  testing_cgetrf_mgpu -M %d -N %d -NGPU %d\n\n", (int) M, (int) N, (int) num_gpus0);
        else {
            printf("\nError: \n");
            printf("  (m, n, num_gpus)=(%d, %d, %d) must be positive.\n\n", (int) M, (int) N, (int) num_gpus0);
            exit(1);
        }
    } else {
        M = N = size[n_size-1];
        printf("\nDefault: \n");
        printf("  testing_cgetrf_mgpu -M %d -N %d -NGPU %d\n\n", (int) M, (int) N, (int) num_gpus0);
    }

    ldda   = ((M+31)/32)*32;
    maxn   = ((N+31)/32)*32;
    n2     = M * N;
    min_mn = min(M, N);
    nb     = magma_get_cgetrf_nb(M);
    num_gpus = num_gpus0;

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
        printf("device i: %d\n", devices[i]);
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
    TESTING_MALLOC_CPU( ipiv, magma_int_t,        min_mn );
    TESTING_MALLOC_CPU( h_A,  magmaFloatComplex, n2     );
    TESTING_MALLOC_CPU( h_R,  magmaFloatComplex, n2     );
    /* allocate device memory, assuming fixed nb and num_gpus */
    for(i=0; i<num_gpus; i++){
          n_local = ((N/nb)/num_gpus)*nb;
          if (i < (N/nb)%num_gpus)
            n_local += nb;
          else if (i == (N/nb)%num_gpus)
            n_local += N%nb;
          ldn_local = ((n_local+31)/32)*32;
      //TESTING_MALLOC_DEV( d_lA[i], magmaFloatComplex, ldda*n_local );
      TESTING_MALLOC_DEV( d_lA[i], magmaFloatComplex, ldda*ldn_local );
    }
    nb0 = nb;

    printf("  M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)  ||PA-LU||/(||A||*N)\n");
    printf("======================================================================\n");
    for(i=0; i<n_size; i++){
        if (flag == 0){
              M = N = size[i];
        }
            min_mn= min(M, N);
            lda   = M;
            n2    = lda*N;
            ldda  = ((M+31)/32)*32;
            gflops = FLOPS( (float)M, (float)N ) *1e-9;

        /* Initialize the matrix */
        lapackf77_clarnv( &ione, ISEED, &n2, h_A );
        lapackf77_clacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

       /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_cgetrf(&M, &N, h_A, &lda, ipiv, &info);
        cpu_time = magma_wtime() - cpu_time;
        if (info < 0) {
            printf("Argument %d of cgetrf had an illegal value.\n", (int) -info);
            break;
        } else if (info != 0 ) {
            printf("cgetrf returned info=%d.\n", (int) info);
            break;
        }
        cpu_perf = gflops / cpu_time;
        lapackf77_clacpy( MagmaUpperLowerStr, &M, &N, h_R, &lda, h_A, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        /* == distributing the matrix == */
        nb = magma_get_cgetrf_nb(M);
        
        if( nb != nb0 ) {
            printf( " different nb used for memory-allocation (%d vs. %d)\n", (int) nb, (int) nb0 );
        }
        
        if( num_gpus0 > N/nb ) {
            num_gpus = N/nb;
            if( N%nb != 0 ) num_gpus ++;
                printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) num_gpus );
        } else {
            num_gpus = num_gpus0;
        }

        for(int j=0; j<N; j+=nb){
            k = (j/nb)%num_gpus;
            nk = min(nb, N-j);
            magma_csetmatrix( M, nk, 
                              &h_R[j*lda], 0, lda, 
                              d_lA[k], j/(nb*num_gpus)*nb*ldda, ldda,
                              queues[2*k]);
        }

        // warm-up
        magma_cgetrf_mgpu( num_gpus, M, N, d_lA, 0, ldda, ipiv, &info, queues);
        
        for(int j=0; j<N; j+=nb){
            k = (j/nb)%num_gpus;
            nk = min(nb, N-j);
            magma_csetmatrix( M, nk, 
                              &h_R[j*lda], 0, lda, 
                              d_lA[k], j/(nb*num_gpus)*nb*ldda, ldda,
                              queues[2*k]);
        }
        /* == calling MAGMA with multiple GPUs == */
        gpu_time = magma_wtime();
        magma_cgetrf_mgpu( num_gpus, M, N, d_lA, 0, ldda, ipiv, &info, queues);
        gpu_time = magma_wtime() - gpu_time;
        gpu_perf = gflops / gpu_time;
        if (info < 0) {
            printf("Argument %d of magma_cgetrf_mgpu had an illegal value.\n", (int) -info);
            break;
        } else if (info != 0 ) {
            printf("magma_cgetrf_mgpu returned info=%d.\n", (int) info);
            break;
        }
        /* == download the matrix from GPUs == */
        for(int j=0; j<N; j+=nb){
            k = (j/nb)%num_gpus;
            nk = min(nb, N-j);
            magma_cgetmatrix( M, nk, 
                              d_lA[k], j/(nb*num_gpus)*nb*ldda, ldda, 
                              &h_R[j*lda], 0, lda, queues[2*k] );
        }
        /* =====================================================================
           Check the factorization
           =================================================================== */
/*
        magma_cprint(M, N, h_R, lda);
        printf("[\n");
        for(int kk=0; kk<min_mn;kk++){
            printf("%d ", ipiv[kk]);
        }
        printf("]\n");
*/        
        error = get_LU_error(M, N, h_A, lda, h_R, ipiv);
        
        printf("%5d %5d  %6.2f (%6.2f)        %6.2f (%6.2f)        %e\n",
               (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error);

        if (flag != 0)
            break;
    }

    /* Memory clean up */
    TESTING_FREE_PIN( ipiv );
    TESTING_FREE_PIN( h_A );
    TESTING_FREE_PIN( h_R );
    for(i=0; i<num_gpus; i++){
        TESTING_FREE_DEV( d_lA[i] );
        magma_queue_destroy( queues[2*i]   );
        magma_queue_destroy( queues[2*i+1] );
    }

    /* Shutdown */
    magma_finalize();
}
