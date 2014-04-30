/*
    -- clMAGMA (version 1.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2014

       @generated from testing_dgeev.cpp normal d -> s, Fri Jan 10 15:51:20 2014

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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgeev
*/

#define magma_abs(x) ((x)>=0.? (x): (-1.*(x)))

extern "C" float magma_slapy2(float x, float y){
  float ret_val, d__1;
  float w, z__, xabs, yabs;

  xabs = magma_abs(x);
  yabs = magma_abs(y);
  w    = fmax(xabs,yabs);
  z__  = fmin(xabs,yabs);
  if (z__ == 0.) {
    ret_val = w;
  } else {
    d__1 = z__ / w;
    ret_val = w * sqrt(d__1 * d__1 + 1.);
  }
  return ret_val;
}

int main( int argc, char** argv)
{
    real_Double_t gpu_time, cpu_time;
    float *h_A, *h_R, *VL, *VR, *h_work, *w1, *w2;
    float *w1i, *w2i;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float matnorm, tnrm, result[8];

    /* Matrix size */
    magma_int_t N=0, n2, lda, nb, lwork;
    magma_int_t size[8] = {1024,2048,3072,4032,5184,6016,7040,8064};

    magma_int_t i, j, info, checkres, once = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_vec_t jobl = MagmaVec;
    magma_vec_t jobr = MagmaVec;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                N = atoi(argv[++i]);
                once = 1;
            }
            else if (strcmp("-LN", argv[i])==0)
                jobl = MagmaNoVec;
            else if (strcmp("-LV", argv[i])==0)
                jobl = MagmaVec;
            else if (strcmp("-RN", argv[i])==0)
                jobr = MagmaNoVec;
            else if (strcmp("-RV", argv[i])==0)
                jobr = MagmaVec;
        }
        if ( N > 0 )
            printf("  testing_sgeev -L[N|V] -R[N|V] -N %d\n\n", (int) N);
        else
        {
            printf("\nUsage: \n");
            printf("  testing_sgeev -L[N|V] -R[N|V] -N %d\n\n", 1024);
            exit(1);
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_sgeev -L[N|V] -R[N|V] -N %d\n\n", 1024);
        N = size[7];
    }

    checkres = getenv("MAGMA_TESTINGS_CHECK") != NULL;

    lda   = N;
    n2    = lda * N;
    nb    = magma_get_sgehrd_nb(N);

    lwork = N*(2+nb);

    // generous workspace - required by sget22
    lwork = max(lwork, N * ( 5 + 2*N));

    TESTING_MALLOC_CPU( w1,  float, N  );
    TESTING_MALLOC_CPU( w2,  float, N  );
    TESTING_MALLOC_CPU( w1i, float, N  );
    TESTING_MALLOC_CPU( w2i, float, N  );
    TESTING_MALLOC_CPU( h_A, float, n2 );
    
    TESTING_MALLOC_PIN( h_R,    float, n2    );
    TESTING_MALLOC_PIN( VL,     float, n2    );
    TESTING_MALLOC_PIN( VR,     float, n2    );
    TESTING_MALLOC_PIN( h_work, float, lwork );

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

    printf("  N     CPU Time(s)    GPU Time(s)     ||R||_F / ||A||_F\n");
    printf("==========================================================\n");
    for(i=0; i<8; i++){
        if ( argc == 1 ){
            N = size[i];
        }
        
        lda = N;
        n2  = lda*N;

        /* Initialize the matrix */
        lapackf77_slarnv( &ione, ISEED, &n2, h_A );
        lapackf77_slacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        // warm-up
        magma_sgeev(jobl, jobr,
                     N, h_R, lda, w1, w1i,
                    VL, lda, VR, lda,
                    h_work, lwork, &info, queue);
        
        lapackf77_slacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
        gpu_time = magma_wtime();
        magma_sgeev(jobl, jobr,
                     N, h_R, lda, w1, w1i,
                    VL, lda, VR, lda,
                    h_work, lwork, &info, queue);

        gpu_time = magma_wtime() - gpu_time;
        if (info < 0)
            printf("Argument %d of magma_sgeev had an illegal value.\n", (int) -info);

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = magma_wtime();
        lapackf77_sgeev(lapack_const(jobl), lapack_const(jobr),
                        &N, h_A, &lda, w2, w2i,
                        VL, &lda, VR, &lda,
                        h_work, &lwork, &info);
        cpu_time = magma_wtime() - cpu_time;
        if (info < 0)
            printf("Argument %d of sgeev had an illegal value.\n", (int) -info);

        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        if ( checkres )
          {
            /* ===================================================================
               Check the result following LAPACK's [zcds]drvev routine.
               The following 7 tests are performed:
               *     (1)     | A * VR - VR * W | / ( n |A| )
               *
               *       Here VR is the matrix of unit right eigenvectors.
               *       W is a diagonal matrix with diagonal entries W(j).
               *
               *     (2)     | A**T * VL - VL * W**T | / ( n |A| )
               *
               *       Here VL is the matrix of unit left eigenvectors, A**T is the
               *       ugate-transpose of A, and W is as above.
               *
               *     (3)     | |VR(i)| - 1 |   and whether largest component real
               *
               *       VR(i) denotes the i-th column of VR.
               *
               *     (4)     | |VL(i)| - 1 |   and whether largest component real
               *
               *       VL(i) denotes the i-th column of VL.
               *
               *     (5)     W(full) = W(partial)
               *
               *       W(full) denotes the eigenvalues computed when both VR and VL
               *       are also computed, and W(partial) denotes the eigenvalues
               *       computed when only W, only W and VR, or only W and VL are
               *       computed.
               *
               *     (6)     VR(full) = VR(partial)
               *
               *       VR(full) denotes the right eigenvectors computed when both VR
               *       and VL are computed, and VR(partial) denotes the result
               *       when only VR is computed.
               *
               *     (7)     VL(full) = VL(partial)
               *
               *       VL(full) denotes the left eigenvectors computed when both VR
               *       and VL are also computed, and VL(partial) denotes the result
               *       when only VL is computed.
               ================================================================= */
            
            int jj;
            float ulp, ulpinv, vmx, vrmx, vtst, res[2];

            float *LRE, DUM;
            TESTING_MALLOC_PIN( LRE, float, n2 );

            ulp = lapackf77_slamch( "P" );
            ulpinv = 1./ulp;

            // Initialize RESULT
            for (j = 0; j < 8; j++)
              result[j] = -1.;

            lapackf77_slarnv( &ione, ISEED, &n2, h_A );
            lapackf77_slacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            magma_sgeev(MagmaVec, MagmaVec,
                        N, h_R, lda, w1, w1i,
                        VL, lda, VR, lda,
                        h_work, lwork, &info, queue);

            // Do test 1
            lapackf77_sget22("N", "N", "N", &N, h_A, &lda, VR, &lda, w1, w1i,
                             h_work, res);
            result[0] = res[0];
            result[0] *= ulp;

            // Do test 2
            lapackf77_sget22("T", "N", "T", &N, h_A, &lda, VL, &lda, w1, w1i,
                             h_work, &result[1]);
            result[1] *= ulp;

            // Do test 3
            result[2] = -1.;
            for (j = 0; j < N; ++j) {
              tnrm = 1.;
              if (w1i[j] == 0.)
                tnrm = cblas_snrm2(N, &VR[j * lda], ione);
              else if (w1i[j] > 0.)
                tnrm = magma_slapy2( cblas_snrm2(N, &VR[j    * lda], ione),
                                     cblas_snrm2(N, &VR[(j+1)* lda], ione) );
              
              result[2] = fmax(result[2], fmin(ulpinv, magma_abs(tnrm-1.)/ulp));
              
              if (w1i[j] > 0.)
                {
                  vmx  = vrmx = 0.;
                  for (jj = 0; jj <N; ++jj) {
                    vtst = magma_slapy2( VR[jj+j*lda], VR[jj+(j+1)*lda]);
                    if (vtst > vmx)
                      vmx = vtst;
                    
                    if ( (VR[jj + (j+1)*lda])==0. &&
                         magma_abs( VR[jj+j*lda] ) > vrmx)
                      vrmx = magma_abs( VR[jj+j*lda] );
                  }
                  if (vrmx / vmx < 1. - ulp * 2.)
                    result[2] = ulpinv;
                }
            }
            result[2] *= ulp;

            // Do test 4
            result[3] = -1.;
            for (j = 0; j < N; ++j) {
              tnrm = 1.;
              if (w1i[j] == 0.)
                tnrm = cblas_snrm2(N, &VL[j * lda], ione);
              else if (w1i[j] > 0.)
                tnrm = magma_slapy2( cblas_snrm2(N, &VL[j    * lda], ione),
                                     cblas_snrm2(N, &VL[(j+1)* lda], ione) );

              result[3] = fmax(result[3], fmin(ulpinv, magma_abs(tnrm-1.)/ulp));

              if (w1i[j] > 0.)
                {
                  vmx  = vrmx = 0.;
                  for (jj = 0; jj <N; ++jj) {
                    vtst = magma_slapy2( VL[jj+j*lda], VL[jj+(j+1)*lda]);
                    if (vtst > vmx)
                      vmx = vtst;

                    if ( (VL[jj + (j+1)*lda])==0. &&
                         magma_abs( VL[jj+j*lda]) > vrmx)
                      vrmx = magma_abs( VL[jj+j*lda] );
                  }
                  if (vrmx / vmx < 1. - ulp * 2.)
                    result[3] = ulpinv;
                }
            }
            result[3] *= ulp;

            // Compute eigenvalues only, and test them
            lapackf77_slacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            magma_sgeev(MagmaNoVec, MagmaNoVec,
                        N, h_R, lda, w2, w2i,
                        &DUM, 1, &DUM, 1,
                        h_work, lwork, &info, queue);

            if (info != 0) {
              result[0] = ulpinv;
             
              info = abs(info);
              printf("Info = %d fo case N, N\n", (int) info);
            }

            // Do test 5
            result[4] = 1;
            for (j = 0; j < N; ++j)
              if ( w1[j] != w2[j] || w1i[j] != w2i[j] )
                result[4] = 0;
            //if (result[4] == 0) printf("test 5 failed with N N\n");

            // Compute eigenvalues and right eigenvectors, and test them
            lapackf77_slacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            magma_sgeev(MagmaNoVec, MagmaVec,
                        N, h_R, lda, w2, w2i,
                        &DUM, 1, LRE, lda,
                        h_work, lwork, &info, queue);

            if (info != 0) {
              result[0] = ulpinv;
              
              info = abs(info);
              printf("Info = %d fo case N, V\n", (int) info);
            }

            // Do test 5 again
            result[4] = 1;
            for (j = 0; j < N; ++j)
              if ( w1[j] != w2[j] || w1i[j] != w2i[j] )
                result[4] = 0;
            //if (result[4] == 0) printf("test 5 failed with N V\n");

            // Do test 6
            result[5] = 1;
            for (j = 0; j < N; ++j)
              for (jj = 0; jj < N; ++jj)
                if ( VR[j+jj*lda] != LRE[j+jj*lda] )
                  result[5] = 0;
 
            // Compute eigenvalues and left eigenvectors, and test them
            lapackf77_slacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            magma_sgeev(MagmaVec, MagmaNoVec,
                        N, h_R, lda, w2, w2i,
                        LRE, lda, &DUM, 1,
                        h_work, lwork, &info, queue);

            if (info != 0) {
              result[0] = ulpinv;

              info = abs(info);
              printf("Info = %d fo case V, N\n", (int) info);
            }

            // Do test 5 again
            result[4] = 1;
            for (j = 0; j < N; ++j)
              if ( w1[j] != w2[j] || w1i[j] != w2i[j] )
                result[4] = 0;
            //if (result[4] == 0) printf("test 5 failed with V N\n");
            
            // Do test 7
            result[6] = 1;
            for (j = 0; j < N; ++j)
              for (jj = 0; jj < N; ++jj)
                if ( VL[j+jj*lda] != LRE[j+jj*lda] )
                  result[6] = 0;
            
            printf("Test 1: | A * VR - VR * W | / ( n |A| ) = %e\n", result[0]);
            printf("Test 2: | A'* VL - VL * W'| / ( n |A| ) = %e\n", result[1]);
            printf("Test 3: |  |VR(i)| - 1    |             = %e\n", result[2]);
            printf("Test 4: |  |VL(i)| - 1    |             = %e\n", result[3]);
            printf("Test 5:   W (full)  ==  W (partial)     = %f\n", result[4]);
            printf("Test 6:  VR (full)  == VR (partial)     = %f\n", result[5]);
            printf("Test 7:  VL (full)  == VL (partial)     = %f\n", result[6]);

            //====================================================================

            matnorm = lapackf77_slange("f", &N, &ione, w1, &N, h_work);
            blasf77_saxpy(&N, &c_neg_one, w1, &ione, w2, &ione);

            result[7] = lapackf77_slange("f", &N, &ione, w2, &N, h_work) / matnorm;

            printf("%5d     %6.2f         %6.2f         %e\n",
                   (int) N, cpu_time, gpu_time, result[7]);

            TESTING_FREE_PIN( LRE );
          }
        else
          {
            printf("%5d     %6.2f         %6.2f\n",
                   (int) N, cpu_time, gpu_time);
          }

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE_CPU( w1  );
    TESTING_FREE_CPU( w2  );
    TESTING_FREE_CPU( w1i );
    TESTING_FREE_CPU( w2i );
    TESTING_FREE_CPU( h_A );
    TESTING_FREE_PIN( h_R );
    TESTING_FREE_PIN( VL  );
    TESTING_FREE_PIN( VR  );
    TESTING_FREE_PIN( h_work );

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
}
