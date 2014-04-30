/*
 *  -- clMAGMA (version 1.1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date January 2014
 *
 * @generated from testing_zgetrf_msub.cpp normal z -> c, Fri Jan 10 15:51:20 2014
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

void init_matrix( int m, int n, magmaFloatComplex *h_A, magma_int_t lda )
{
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    lapackf77_clarnv( &ione, ISEED, &n2, h_A );
}

float get_residual(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    if ( m != n ) {
        printf( "\nERROR: residual check defined only for square matrices\n" );
        return -1;
    }

    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t ione = 1;

    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,2};
    magma_int_t info = 0;
    magmaFloatComplex *x, *b;

    // initialize RHS
    TESTING_MALLOC_CPU( x, magmaFloatComplex, n );
    TESTING_MALLOC_CPU( b, magmaFloatComplex, n );
    lapackf77_clarnv( &ione, ISEED, &n, b );
    blasf77_ccopy( &n, b, &ione, x, &ione );

    // solve Ax = b
    lapackf77_cgetrs( "Notrans", &n, &ione, A, &lda, ipiv, x, &n, &info );
    if ( info != 0 )
        printf( "lapackf77_cgetrs returned error %d\n", info );

    // reset to original A
    init_matrix( m, n, A, lda );

    // compute r = Ax - b, saved in b
    blasf77_cgemv( "Notrans", &m, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );

    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_clange( "F", &m, &n, A, &lda, work );
    norm_r = lapackf77_clange( "F", &n, &ione, b, &n, work );
    norm_x = lapackf77_clange( "F", &n, &ione, x, &n, work );

    TESTING_FREE_CPU( x );
    TESTING_FREE_CPU( b );

    return norm_r / (n * norm_A * norm_x);
}

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
    magmaFloatComplex *h_A, *h_R, *h_P;
    magmaFloatComplex_ptr d_lA[MagmaMaxSubs*MagmaMaxGPUs];
    magma_int_t     *ipiv, check = 0;

    /* Matrix size */
    magma_int_t i, j, k, info, maxm, min_mn, nb, nk, trans = MagmaNoTrans;
    magma_int_t M = 0, N = 0, flag = 0, n2, lda, ldda, num_gpus = 1, num_subs = 1, n_local;
    magma_int_t size[8] = {1000,2000,3000,4000,5000,6000,7000,8000};
    magma_int_t n_size = 8;

    /* Initialize */
    magma_queue_t  queues[MagmaMaxGPUs * 2];
    magma_device_t devices[MagmaMaxGPUs];
    int num = 0;
    magma_err_t err;

    /* command-line arguments */
    if (argc != 1){
        for (i = 1; i<argc; i++) {
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
            } else if (strcmp("-TRANS", argv[i])==0) {
                trans = MagmaTrans;
            } else if (strcmp("-check", argv[i])==0) {
                check = atoi(argv[++i]);
            }
        }
    }
    if (flag != 0) {
        if( num_gpus <= 0 || num_gpus > MagmaMaxGPUs || num_subs <= 0 || num_subs > MagmaMaxSubs ) {
            printf("\nError: \n");
            printf("  (MaxGPUs, MaxSubs)=(%d, %d).\n\n", (int) MagmaMaxGPUs, (int) MagmaMaxSubs);
            exit(1);
        } else if (M < 0 || N < 0 ) {
            printf("\nError: \n");
            printf("  (m, n, num_gpus, num_subs)=(%d, %d, %d, %d) must be positive.\n\n", (int) M, (int) N, (int) num_gpus, (int) num_subs);
            exit(1);
        } else {
            printf("\n  testing_cgetrf_mgpu -M %d -N %d -NGPU %d -NSUB %d -check %d\n\n", (int) M, (int) N, (int) num_gpus, (int) num_subs, (int) check);
        }
    } else {
        M = N = size[n_size-1];
        printf("\nDefault: \n");
        printf("  testing_cgetrf_mgpu -M %d -N %d -NGPU %d -NSUB %d -check %d\n\n", (int) M, (int) N, (int) num_gpus, (int) num_subs, (int) check);
    }

    /* initialize MAGMA */
    magma_init();
    err = magma_get_devices( devices, MagmaMaxGPUs, &num );
    if (err != 0 || num < 1) {
        fprintf( stderr, "magma_get_devices failed: %d\n", err );
        exit(-1);
    }
    printf( "%d devices initialized (num_gpus=%d, num_subs=%d)\n\n",num,num_gpus,num_subs );

    /* start testing */
    if (check == 2) {
        printf("  M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)  ||b-Ax||/(||A||||x||*N)\n");
    } else {
        printf("  M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)  ||PA-LU||/(||A||*N)\n");
    }
    printf("===========================================================================\n");
    for (i=0; i<n_size; i++) {
        if (flag == 0){
            M = N = size[i];
        }
        min_mn = min(M, N);
        maxm   = 32*((M+31)/32);
        lda    = M;
        n2     = lda*N;
        nb     = magma_get_cgetrf_nb(M);
        gflops = FLOPS_CGETRF( M, N ) / 1e9;
        if (num_subs*num_gpus > N/nb) {
            printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) num_gpus );
            exit( 0 );
        } 

        /* Create queues */
        for (j=0;j<num_gpus;j++) {
            err = magma_queue_create( devices[j%num_gpus], &queues[2*j] );
            if (err != 0) {
                fprintf( stderr, "magma_queue_create failed: %d\n", err );
                exit(-1);
            }
            err = magma_queue_create( devices[j%num_gpus], &queues[2*j+1] );
            if (err != 0) {
                fprintf( stderr, "magma_queue_create failed: %d\n", err );
                exit(-1);
            }
        }

        /* Allocate host memory for the matrix */
        TESTING_MALLOC_CPU( ipiv, magma_int_t, min_mn );
        TESTING_MALLOC_CPU( h_P, magmaFloatComplex, lda*nb );
        TESTING_MALLOC_CPU( h_R, magmaFloatComplex, n2 );
        /* Allocate device memory */
        if (trans == MagmaNoTrans) {
            ldda = N/nb;                     /* number of block columns         */
            ldda = ldda/(num_gpus*num_subs); /* number of block columns per GPU */
            ldda = nb*ldda;                  /* number of columns per GPU       */
            if (ldda * num_gpus*num_subs < N) {
                /* left over */
                if (N-ldda*num_gpus*num_subs >= nb) {
                    ldda += nb;
                } else {
                    ldda += (N-ldda*num_gpus*num_subs)%nb;
                }
            }
            ldda = ((ldda+31)/32)*32; /* make it a multiple of 32 */
            for (j=0; j<num_subs * num_gpus; j++) {
                TESTING_MALLOC_DEV( d_lA[j], magmaFloatComplex, ldda*maxm );
            }
        } else {
            ldda = ((M+31)/32)*32;
            for (j=0; j<num_subs * num_gpus; j++) {
                n_local = ((N/nb)/(num_subs*num_gpus))*nb;
                if (j < (N/nb)%(num_subs*num_gpus)) {
                    n_local += nb;
                } else if (j == (N/nb)%(num_subs*num_gpus)) {
                    n_local += N%nb;
                }
                TESTING_MALLOC_DEV( d_lA[j], magmaFloatComplex, ldda*n_local );
            }
        }

        /* Initialize the matrix */
        init_matrix( M, N, h_R, lda );

       /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_cgetrf(&M, &N, h_R, &lda, ipiv, &info);
        cpu_time = magma_wtime() - cpu_time;
        if (info < 0) {
            printf("Argument %d of cgetrf had an illegal value.\n", (int) -info);
            break;
        } else if (info != 0 ) {
            printf("cgetrf returned info=%d.\n", (int) info);
            break;
        }
        cpu_perf = gflops / cpu_time;

        /* ====================================================================
           Call MAGMA for warmup
           =================================================================== */
        /* == distributing the matrix == */
        init_matrix( M, N, h_R, lda );
        if (trans == MagmaNoTrans) {
            for (j=0; j<N; j+=nb) {
                k = (j/nb)%(num_subs*num_gpus);
                nk = min(nb, N-j);

                /* transpose on CPU, then copy to GPU */
                int ii, jj;
                for( ii=0; ii<M; ii++ ) {
                    for( jj=0; jj<nk; jj++ ) {
                        h_P[jj+ii*nk] = h_R[ii+jj*lda];
                    }
                }
                magma_csetmatrix( nk, M,
                                  h_P, 0, nk, 
                                  d_lA[k], j/(nb*num_subs*num_gpus)*nb, ldda,
                                  queues[2*(k%num_gpus)]);
            }
        } else {
            ldda = ((M+31)/32)*32;
            for(j=0; j<N; j+=nb){
                k = (j/nb)%(num_subs*num_gpus);
                nk = min(nb, N-j);
                magma_csetmatrix( M, nk, 
                                  h_R, j*lda, lda, 
                                  d_lA[k], j/(nb*num_subs*num_gpus)*nb*ldda, ldda,
                                  queues[2*(k%num_gpus)]);
            }
        }
        /* warm-up */
        magma_cgetrf_msub( trans, num_subs, num_gpus, M, N, d_lA, 0, ldda, ipiv, &info, queues);

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        init_matrix( M, N, h_R, lda );
        if (trans == MagmaNoTrans) {
            for(j=0; j<N; j+=nb){
                k = (j/nb)%(num_subs*num_gpus);
                nk = min(nb, N-j);

                /* transpose on CPU, then copy to GPU */
                int ii,jj;
                for (ii=0; ii<M; ii++) {
                    for( jj=0; jj<nk; jj++ ) {
                        h_P[jj+ii*nk] = h_R[j*lda + ii+jj*lda];
                    }
                }
                magma_csetmatrix( nk, M,
                                  h_P, 0, nk, 
                                  d_lA[k], j/(nb*num_subs*num_gpus)*nb, ldda,
                                  queues[2*(k%num_gpus)]);
            }
        } else {
            ldda = ((M+31)/32)*32;
            for (j=0; j<N; j+=nb) {
                k = (j/nb)%(num_subs*num_gpus);
                nk = min(nb, N-j);
                magma_csetmatrix( M, nk, 
                                  h_R, j*lda, lda, 
                                  d_lA[k], j/(nb*num_subs*num_gpus)*nb*ldda, ldda,
                                  queues[2*(k%num_gpus)]);
            }
        }
        /* == calling MAGMA with multiple GPUs == */
        gpu_time = magma_wtime();
        magma_cgetrf_msub( trans, num_subs, num_gpus, M, N, d_lA, 0, ldda, ipiv, &info, queues);
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
        if (trans == MagmaNoTrans) {
            for (j=0; j<N; j+=nb) {
                k = (j/nb)%(num_subs*num_gpus);
                nk = min(nb, N-j);

                /* copy to CPU and then transpose */
                magma_cgetmatrix( nk, M, 
                                  d_lA[k], j/(nb*num_subs*num_gpus)*nb, ldda, 
                                  h_P, 0, nk, queues[2*(k%num_gpus)] );
                int ii, jj;
                for (ii=0; ii<M; ii++) {
                    for( jj=0; jj<nk; jj++ ) {
                        h_R[j*lda + ii+jj*lda] = h_P[jj+ii*nk];
                    }
                }
            }
        } else {
            for (j=0; j<N; j+=nb) {
                k = (j/nb)%(num_subs*num_gpus);
                nk = min(nb, N-j);
                magma_cgetmatrix( M, nk, 
                                  d_lA[k], j/(nb*num_subs*num_gpus)*nb*ldda, ldda, 
                                  h_R, j*lda, lda, queues[2*(k%num_gpus)] );
            }
        }

        /* =====================================================================
           Check the factorization
           =================================================================== */
        if (check == 1) {
            TESTING_MALLOC_CPU( h_A, magmaFloatComplex, n2 );
            init_matrix( M, N, h_A, lda );
            error = get_LU_error(M, N, h_A, lda, h_R, ipiv);
            TESTING_FREE_PIN( h_A );
        
            printf("%5d %5d  %6.2f (%6.2f)        %6.2f (%6.2f)        %e\n",
                   (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error);
        } else if (check == 2) {
            error = get_residual(M, N, h_R, lda, ipiv );
            printf("%5d %5d  %6.2f (%6.2f)        %6.2f (%6.2f)        %e\n",
                   (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error);
        } else {
            printf("%5d %5d  %6.2f (%6.2f)        %6.2f (%6.2f)        ---\n",
                   (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time);
        }
        /* Memory clean up */
        for (j=0; j<num_gpus; j++) {
            magma_queue_destroy(queues[2*j]);
            magma_queue_destroy(queues[2*j+1]);
            for (k=0; k<num_subs; k++) {
                TESTING_FREE_DEV( d_lA[j*num_subs+k] );
            }
        }
        TESTING_FREE_PIN( ipiv );
        TESTING_FREE_PIN( h_P );
        TESTING_FREE_PIN( h_R );

        if (flag != 0)
            break;
    }

    /* Shutdown */
    magma_finalize();
    return 0;
}
