/*
 *  -- clMAGMA (version 1.1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date January 2014
 *
 * @generated from testing_zgebrd.cpp normal z -> c, Fri Jan 10 15:51:20 2014
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
#define FLOPS(m, n) ( 6. * FMULS_GEBRD(m, n) + 2. * FADDS_GEBRD(m, n))
#else
#define FLOPS(m, n) (      FMULS_GEBRD(m, n) +      FADDS_GEBRD(m, n))
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgebrd
*/
int main( int argc, char** argv)
{
    real_Double_t      eps, gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    magmaFloatComplex *h_A, *h_Q, *h_PT, *h_work, *chkwork;
    magmaFloatComplex *taup, *tauq;
    float          *diag, *offdiag, *rwork;
    float           result[3] = {0., 0., 0.};

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2, lda, lhwork, lchkwork;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    magma_int_t i, info, minmn, nb, uselapack, checkres;
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
        if (N>0 && M>0)
            printf("  testing_cgebrd -M %d -N %d\n\n", (int) M, (int) N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_cgebrd -M %d -N %d\n\n", 1024, 1024);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_cgebrd -M %d -N %d\n\n", 1024, 1024);
        M = N = size[9];
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

    uselapack = getenv("MAGMA_USE_LAPACK") != NULL;
    checkres  = getenv("MAGMA_TESTINGS_CHECK") != NULL;

    eps = lapackf77_slamch( "E" );
    lda = M;
    n2  = lda * N;
    nb  = magma_get_cgebrd_nb(N);
    minmn = min(M, N);

    /* Allocate host memory for the matrix */
    TESTING_MALLOC_CPU( h_A,     magmaFloatComplex, lda*N );
    TESTING_MALLOC_CPU( tauq,    magmaFloatComplex, minmn  );
    TESTING_MALLOC_CPU( taup,    magmaFloatComplex, minmn  );
    TESTING_MALLOC_CPU( diag,    float, minmn   );
    TESTING_MALLOC_CPU( offdiag, float, (minmn-1) );
    TESTING_MALLOC_PIN( h_Q, magmaFloatComplex, lda*N );

    lhwork = (M + N)*nb;
    TESTING_MALLOC_PIN( h_work, magmaFloatComplex, lhwork );

    /* To avoid uninitialized variable warning */
    h_PT    = NULL;
    chkwork = NULL;
    rwork   = NULL;

    if ( checkres ) {
        lchkwork = max(minmn * nb, M+N);
        /* For optimal performance in cunt01 */
        lchkwork = max(lchkwork, minmn*minmn);
        TESTING_MALLOC_CPU( h_PT,    magmaFloatComplex, lda*N   );
        TESTING_MALLOC_CPU( chkwork, magmaFloatComplex, lchkwork );
#if defined(PRECISION_z) || defined(PRECISION_c)
        TESTING_MALLOC_CPU( rwork, float, 5*minmn );
#endif
    }

    printf("  M    N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("==================================================================\n");
    for(i=0; i<10; i++){
        if (argc == 1) {
            M = N = size[i];
        }
        minmn = min(M, N);
        lda   = M;
        n2    = lda*N;
        lhwork   = (M + N)*nb;
        lchkwork = max(minmn * nb, M+N);
        /* For optimal performance in cunt01 */
        lchkwork = max(lchkwork, minmn*minmn);
        gflops = FLOPS( (float)M, (float)N ) * 1e-9;

        /* Initialize the matrices */
        lapackf77_clarnv( &ione, ISEED, &n2, h_A );
        lapackf77_clacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_Q, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        gpu_time = magma_wtime();
        if ( uselapack ) {
            lapackf77_cgebrd( &M, &N, h_Q, &lda,
                              diag, offdiag, tauq, taup,
                              h_work, &lhwork, &info);
        } else {
            magma_cgebrd( M, N, h_Q, lda,
                          diag, offdiag, tauq, taup,
                          h_work, lhwork, &info, queue );
        }
        gpu_time = magma_wtime() - gpu_time;
        if ( info < 0 )
            printf("Argument %d of lapackf77_cgebrd|magma_cgebrd had an illegal value\n", (int) -info);

        gpu_perf = gflops / gpu_time;

        /* =====================================================================
           Check the factorization
           =================================================================== */
        if ( checkres ) {
            lapackf77_clacpy(MagmaUpperLowerStr, &M, &N, h_Q, &lda, h_PT, &lda);
            
            // generate Q & P'
            lapackf77_cungbr("Q", &M, &minmn, &N, h_Q,  &lda, tauq, chkwork, &lchkwork, &info);
            if ( info < 0 )
              printf("Argument %d of lapackf77_cungbr had an illegal value\n", (int) -info);
            lapackf77_cungbr("P", &minmn, &N, &M, h_PT, &lda, taup, chkwork, &lchkwork, &info);
            if ( info < 0 )
              printf("Argument %d of lapackf77_cungbr (2) had an illegal value\n", (int) -info);
            
            // Test 1:  Check the decomposition A := Q * B * PT
            //      2:  Check the orthogonality of Q
            //      3:  Check the orthogonality of PT
#if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_cbdt01(&M, &N, &ione,
                             h_A, &lda, h_Q, &lda,
                             diag, offdiag, h_PT, &lda,
                             chkwork, rwork, &result[0]);
            lapackf77_cunt01("Columns", &M, &minmn, h_Q,  &lda, chkwork, &lchkwork, rwork, &result[1]);
            lapackf77_cunt01("Rows",    &minmn, &N, h_PT, &lda, chkwork, &lchkwork, rwork, &result[2]);
#else
            lapackf77_cbdt01(&M, &N, &ione,
                             h_A, &lda, h_Q, &lda,
                             diag, offdiag, h_PT, &lda,
                             chkwork, &result[0]);
            lapackf77_cunt01("Columns", &M, &minmn, h_Q,  &lda, chkwork, &lchkwork, &result[1]);
            lapackf77_cunt01("Rows",    &minmn, &N, h_PT, &lda, chkwork, &lchkwork, &result[2]);
#endif
        }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_cgebrd(&M, &N, h_A, &lda,
                         diag, offdiag, tauq, taup,
                         h_work, &lhwork, &info);
        cpu_time = magma_wtime() - cpu_time;

        if (info < 0)
            printf("Argument %d of lapackf77_cgebrd had an illegal value.\n", (int) -info);

        cpu_perf = gflops / cpu_time;

        /* =====================================================================
           Print performance and error.
           =================================================================== */
        if ( checkres ) {
            printf("%5d %5d   %6.2f        %6.2f       %4.2e %4.2e %4.2e\n",
                   (int) M, (int) N, cpu_perf, gpu_perf,
                   result[0]*eps, result[1]*eps, result[2]*eps );
        } else {
            printf("%5d %5d   %6.2f        %6.2f\n",
                   (int) M, (int) N, cpu_perf, gpu_perf );
        }

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE_CPU( h_A );
    TESTING_FREE_CPU( tauq );
    TESTING_FREE_CPU( taup );
    TESTING_FREE_CPU( diag );
    TESTING_FREE_CPU( offdiag );
    TESTING_FREE_PIN( h_Q );
    TESTING_FREE_PIN( h_work );

    if ( checkres ) {
        TESTING_FREE_CPU( h_PT );
        TESTING_FREE_CPU( chkwork );
#if defined(PRECISION_z) || defined(PRECISION_c)
        TESTING_FREE_CPU( rwork );
#endif
    }

    magma_queue_destroy( queue );
    magma_finalize();
}
