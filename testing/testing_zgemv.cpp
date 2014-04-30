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
#define FLOPS(m, n) ( 6. * FMULS_GEMV(m, n) + 2. * FADDS_GEMV(m, n))
#else
#define FLOPS(m, n) (      FMULS_GEMV(m, n) +      FADDS_GEMV(m, n))
#endif

int main(int argc, char **argv)
{
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    double error, work[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    magma_int_t i, lda, Xm, Ym;
    magma_int_t M, M0 = 0;
    magma_int_t N, N0 = 0;
    magma_int_t szeA, szeX, szeY;
    magma_int_t istart = 64;
    magma_int_t iend = 7070;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_trans_t trans = MagmaNoTrans;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(1., 0.); // MAGMA_Z_MAKE(  1.5, -2.3 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE(0., 0.); // MAGMA_Z_MAKE( -0.6,  0.8 );
    magmaDoubleComplex *A, *X, *Y, *Ymagma, *Ycpu;
    magmaDoubleComplex_ptr dA, dX, dY;
        
    if (argc != 1){
        for(i=1; i<argc; i++){
            if ( strcmp("-n", argv[i]) == 0 ){
                N0 = atoi(argv[++i]);
            }
            else if ( strcmp("-m", argv[i]) == 0 ){
                M0 = atoi(argv[++i]);
            }
            else if (strcmp("-N", argv[i])==0){
                trans = MagmaNoTrans;
            }
            else if (strcmp("-T", argv[i])==0){
                trans = MagmaTrans;
            }
#if defined(PRECISION_z) || defined(PRECISION_c)
            else if (strcmp("-C", argv[i])==0){
                trans = MagmaConjTrans;
            }
#endif
        }
    }

    if ( (M0 != 0) && (N0 != 0) )
        iend = istart + 1;

    M = N = iend;
    if ( M0 != 0 ) M = M0;
    if ( N0 != 0 ) N = N0;

    if( trans == MagmaNoTrans ) {
        Xm = N;
        Ym = M;
    }  else {
        Xm = M;
        Ym = N;
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
    
    lda = ((M+31)/32)*32;
    
    szeA = lda*N;
    szeX = incx*Xm;
    szeY = incy*Ym;
      
    TESTING_MALLOC_PIN( A, magmaDoubleComplex, szeA );
    TESTING_MALLOC_PIN( X, magmaDoubleComplex, szeX );
    TESTING_MALLOC_PIN( Y, magmaDoubleComplex, szeY );
    TESTING_MALLOC_PIN( Ymagma, magmaDoubleComplex, szeY );
    TESTING_MALLOC_PIN( Ycpu, magmaDoubleComplex, szeY );

    TESTING_MALLOC_DEV( dA, magmaDoubleComplex, szeA );
    TESTING_MALLOC_DEV( dX, magmaDoubleComplex, szeX );
    TESTING_MALLOC_DEV( dY, magmaDoubleComplex, szeY );

    /* Initialize the matrix */
    lapackf77_zlarnv( &ione, ISEED, &szeA, A );
    lapackf77_zlarnv( &ione, ISEED, &szeX, X );
    lapackf77_zlarnv( &ione, ISEED, &szeY, Y );

    printf("\nUsage: \n");
    printf("  testing_zgemv [-N|T|C] [-m %d] [-n %d]\n\n", 1024, 1024);

    printf("  M     N    CPU GFlop/s (sec)   GPU GFlop/s (sec)   error\n");
    printf("======================================================================\n");
    for( i=istart; i < iend; i = (int)((i+1)*1.1) )
    {
        M = N = i;
        if ( M0 != 0 ) M = M0;
        if ( N0 != 0 ) N = N0;

        if( trans == MagmaNoTrans ) {
            Xm = N;
            Ym = M;
        }  else {
            Xm = M;
            Ym = N;
        }
         
        lda = ((M+31)/32)*32;
        gflops = FLOPS( (double)M, (double)N ) / 1e9;

        /* =====================================================================
           Performs operation using OPENCL-BLAS
           =================================================================== */
        magma_zsetmatrix( M, N, A, 0, lda, dA, 0, lda, queue );
        magma_zsetvector( Xm, X, 0, incx, dX, 0, incx, queue );
        magma_zsetvector( Ym, Y, 0, incy, dY, 0, incy, queue );

        /*
         * OPENCL BLAS Version
         */
        // warm-up
        magma_zgemv( trans, M, N, alpha, dA, 0, lda, dX, 0, incx, beta, dY, 0, incy, queue );
        clFinish(queue);

        magma_zsetvector( Ym, Y, 0, incy, dY, 0, incy, queue );
        gpu_time = magma_wtime();
        magma_zgemv( trans, M, N, alpha, dA, 0, lda, dX, 0, incx, beta, dY, 0, incy, queue );
        clFinish(queue);
        gpu_time = magma_wtime() - gpu_time;
        
        magma_zgetvector( Ym, dY, 0, incy, Ymagma, 0, incy, queue );
        
        gpu_perf = gflops / gpu_time;

        /*
         * Blas comparaison
         */
        const char *blastrans = MagmaNoTransStr;
        if ( trans == MagmaConjTrans )
            blastrans = MagmaConjTransStr;
        else if ( trans == MagmaTrans )
            blastrans = MagmaTransStr;
            
        blasf77_zcopy( &Ym, Y, &incy, Ycpu, &incy );
        cpu_time = magma_wtime();
        blasf77_zgemv( blastrans, &M, &N,
                        &alpha, A,       &lda,
                                X,       &incx,
                        &beta,  Ycpu, &incy );
        cpu_time = magma_wtime() - cpu_time;
        cpu_perf = gflops / cpu_time;
            
        blasf77_zaxpy( &Ym, &c_neg_one, Ymagma, &incy, Ycpu, &incy);
        error = lapackf77_zlange( "M", &Ym, &ione, Ycpu, &Ym, work );

        printf("%5d %5d   %6.2f (%6.2f)     %6.2f (%6.2f)       %e\n",
               M, N, cpu_perf, cpu_time, gpu_perf, gpu_time, error/(double)Ym);
    }
    
    /* Free Memory */
    TESTING_FREE_PIN( A );
    TESTING_FREE_PIN( X );
    TESTING_FREE_PIN( Y );
    TESTING_FREE_PIN( Ycpu );
    TESTING_FREE_PIN( Ymagma );

    TESTING_FREE_DEV( dA );
    TESTING_FREE_DEV( dX );
    TESTING_FREE_DEV( dY );

    /* Free device */
    magma_queue_destroy( queue );
    magma_finalize();
}
