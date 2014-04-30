/*
    -- MAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

    @author Stan Tomov

    @generated from testing_zheevd.cpp normal z -> c, Fri Jan 10 15:51:20 2014

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define absv(v1) ((v1)>0? (v1): -(v1))

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cheevd
*/
int main( int argc, char** argv)
{
    real_Double_t gpu_time, cpu_time;

    magmaFloatComplex *h_A, *h_R, *h_work;
    float *rwork, *w1, *w2;
    magma_int_t *iwork;

    /* Matrix size */
    magma_int_t N=0, n2;
    magma_int_t size[8] = {1024,2048,3072,4032,5184,6016,7040,8064};

    magma_int_t i, info;
    magma_int_t ione     = 1, izero = 0;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_uplo_t uplo = MagmaLower;
    magma_vec_t jobz = MagmaVec;

    magma_int_t checkres;
    float result[3], eps = lapackf77_slamch( "E" );

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                N = atoi(argv[++i]);
            }
            else if ( strcmp("-JV", argv[i]) == 0 ) {
                jobz = MagmaVec;
            }
            else if ( strcmp("-JN", argv[i]) == 0 ) {
                jobz = MagmaNoVec;
            }
        }
        if (N>0)
            printf("  testing_cheevd -N %d [-JV] [-JN]\n\n", (int) N);
        else {
            printf("\nUsage: \n");
            printf("  testing_cheevd -N %d [-JV] [-JN]\n\n", (int) N);
            exit(1);
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_cheevd -N %d [-JV] [-JN]\n\n", 1024);
        N = size[7];
    }

    checkres  = getenv("MAGMA_TESTINGS_CHECK") != NULL;
    if ( checkres && jobz == MagmaNoVec ) {
        printf( "Cannot check results when vectors are not computed (jobz='N')\n" );
        checkres = false;
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

    /* Query for workspace sizes */
    magmaFloatComplex aux_work[1];
    float          aux_rwork[1];
    magma_int_t     aux_iwork[1];
    magma_cheevd( jobz, uplo,
                  N, h_R, N, w1,
                  aux_work,  -1,
                  aux_rwork, -1,
                  aux_iwork, -1,
                  &info, queue );
    magma_int_t lwork, lrwork, liwork;
    lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
    lrwork = (magma_int_t) aux_rwork[0];
    liwork = aux_iwork[0];

    /* Allocate host memory for the matrix */
    TESTING_MALLOC_CPU( h_A,    magmaFloatComplex, N*N );
    TESTING_MALLOC_CPU( w1,     float,             N   );
    TESTING_MALLOC_CPU( w2,     float,             N   );
    TESTING_MALLOC_PIN( h_R,    magmaFloatComplex, N*N );
    TESTING_MALLOC_PIN( h_work, magmaFloatComplex, lwork  );
    TESTING_MALLOC_CPU( rwork,  float,             lrwork );
    TESTING_MALLOC_CPU( iwork,  magma_int_t,        liwork );
    
    printf("  N     CPU Time(s)    GPU Time(s) \n");
    printf("===================================\n");
    for(i=0; i<8; i++){
        if (argc==1){
            N = size[i];
        }
        n2 = N*N;

        /* Initialize the matrix */
        lapackf77_clarnv( &ione, ISEED, &n2, h_A );
        for( int i=0; i<N; i++) {
            MAGMA_C_SET2REAL( h_A[i*N+i], MAGMA_C_REAL(h_A[i*N+i]) );
        }
        lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );

        /* warm up run */
        magma_cheevd(jobz, uplo,
                     N, h_R, N, w1,
                     h_work, lwork,
                     rwork, lrwork,
                     iwork, liwork,
                     &info, queue);
        
        lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );

        /* query for optimal workspace sizes */
        magma_cheevd(jobz, uplo,
                     N, h_R, N, w1,
                     h_work, -1,
                     rwork,  -1,
                     iwork,  -1,
                     &info, queue);
        int lwork_save  = lwork;
        int lrwork_save = lrwork;
        int liwork_save = liwork;
        lwork  = min( lwork,  (magma_int_t) MAGMA_C_REAL( h_work[0] ));
        lrwork = min( lrwork, (magma_int_t) rwork[0] );
        liwork = min( liwork, iwork[0] );
        //printf( "lwork %d, query %d, used %d; liwork %d, query %d, used %d\n",
        //        lwork_save,  (magma_int_t) h_work[0], lwork,
        //        liwork_save, iwork[0], liwork );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        gpu_time = magma_wtime();
        magma_cheevd(jobz, uplo,
                     N, h_R, N, w1,
                     h_work, lwork,
                     rwork, lrwork,
                     iwork, liwork,
                     &info, queue);
        gpu_time = magma_wtime() - gpu_time;

        lwork  = lwork_save;
        lrwork = lrwork_save;
        liwork = liwork_save;
        
        if ( checkres ) {
          /* =====================================================================
             Check the results following the LAPACK's [zcds]drvst routine.
             A is factored as A = U S U' and the following 3 tests computed:
             (1)    | A - U S U' | / ( |A| N )
             (2)    | I - U'U | / ( N )
             (3)    | S(with U) - S(w/o U) | / | S |
             =================================================================== */
          float temp1, temp2;
          magmaFloatComplex *tau;

          lapackf77_chet21(&ione, lapack_const(uplo), &N, &izero,
                           h_A, &N,
                           w1, w1,
                           h_R, &N,
                           h_R, &N,
                           tau, h_work, rwork, &result[0]);
          
          lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
          magma_cheevd(MagmaNoVec, uplo,
                       N, h_R, N, w2,
                       h_work, lwork,
                       rwork, lrwork,
                       iwork, liwork,
                       &info, queue);

          temp1 = temp2 = 0;
          for(int j=0; j<N; j++){
            temp1 = max(temp1, absv(w1[j]));
            temp1 = max(temp1, absv(w2[j]));
            temp2 = max(temp2, absv(w1[j]-w2[j]));
          }
          result[2] = temp2 / temp1;
        }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_cheevd(lapack_const(jobz), lapack_const(uplo),
                         &N, h_A, &N, w2,
                         h_work, &lwork,
                         rwork, &lrwork,
                         iwork, &liwork,
                         &info);
        cpu_time = magma_wtime()-cpu_time;
        if (info < 0)
          printf("Argument %d of cheevd had an illegal value.\n", (int) -info);

        /* =====================================================================
           Print execution time
           =================================================================== */
        printf("%5d     %6.2f         %6.2f\n",
               (int) N, cpu_time, gpu_time);
        if ( checkres ){
          printf("Testing the factorization A = U S U' for correctness:\n");
          printf("(1)    | A - U S U' | / (|A| N) = %e\n", result[0]*eps);
          printf("(2)    | I -   U'U  | /  N      = %e\n", result[1]*eps);
          printf("(3)    | S(w/ U)-S(w/o U)|/ |S| = %e\n\n", result[2]);
        }

        if (argc != 1)
            break;
    }
 
    /* Memory clean up */
    TESTING_FREE_CPU( h_A );
    TESTING_FREE_CPU( w1  );
    TESTING_FREE_CPU( w2  );
    TESTING_FREE_CPU( rwork );
    TESTING_FREE_CPU( iwork );
    TESTING_FREE_PIN( h_work );
    TESTING_FREE_PIN( h_R );

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
}
