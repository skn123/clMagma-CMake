/*
 *  -- clMAGMA (version 1.1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date January 2014
 *
 *
 **/

// IWOCL 2013 benchmark

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "assert.h"
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

// globals
extern cl_context gContext;

int main( int argc, char** argv)
{
    real_Double_t t_start, t_end, t_avg, t_max, t_min;

    magmaDoubleComplex_ptr d_A, d_B, d_C;
    int type = 4;
    
    int i;
    if (argc != 1){
        for(i=1; i<argc; i++){
            if ( strcmp("-T", argv[i]) == 0 ){
                type = atoi(argv[++i]);
            }
        }
    }

    if( type == 4){
        printf("Usage:\n");
        printf("testing_benchmark -T 0,    testing kernel launch overhead (using empty kernel)\n");
        printf("testing_benchmark -T 1,    testing kernel launch overhead (using GPU to GPU mem cpy)\n");
        printf("testing_benchmark -T 2,    testing blas kernel launch overhead \n");
        printf("testing_benchmark -T 3,    testing transfer latencies\n");
        exit(0);
    }

    /* Initialize */
    magma_queue_t  queue;
    magma_device_t device[MagmaMaxGPUs];
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

    if (type == 0){
        printf("testing for kernel launch overhead, using empty kernel\n");
        double klatency = 0.0;
        TESTING_MALLOC_DEV( d_C, double, 10 );
        TESTING_MALLOC_DEV( d_A, double, 10 );
        TESTING_MALLOC_DEV( d_B, double, 10 );
        int count = 10000;
        //int count = 400;
        int j;
        magmablas_zempty(queue, d_A, d_B, d_C);
        clFinish(queue);
        int kk;
        t_avg = 0, t_min = 50, t_max = 0;
        for(kk=0;kk<20;kk++){    
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }
                magmablas_zempty(queue, d_A, d_B, d_C);
                clFlush(queue);
            }
            clFinish(queue);
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("#%d: Async Latency: %f us\n", (kk+1), klatency);
        }
        t_avg = t_avg/20;
        printf("Async Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);

        t_avg = 0, t_min = 1000, t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }
                magmablas_zempty(queue, d_A, d_B, d_C);
                clFinish(queue);
            }
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Sync Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Sync Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);

        TESTING_FREE_DEV( d_A );
        TESTING_FREE_DEV( d_B );
        TESTING_FREE_DEV( d_C );
    }


    if (type == 1){
        printf("testing for kernel launch overhead, using GPU to GPU mem copy\n");
        double klatency;
        int count = 10000;
        double* h_A; 
        TESTING_MALLOC_CPU( h_A, double, 100 );
        TESTING_MALLOC_DEV( d_A, double, 1 );
        TESTING_MALLOC_DEV( d_B, double, 1 );
        int j;
        clEnqueueCopyBuffer(queue, d_A, d_A, 0, 0, 1, 0, NULL, NULL);
        clFinish(queue);
        for(j=0;j<count;j++){
            if(j==0){
                t_start = magma_wtime();
            }        
            clEnqueueCopyBuffer(queue, d_A, d_B, 0, 0, 1, 0, NULL, NULL);
            //clEnqueueWriteBuffer(queue, d_A, CL_FALSE, 0, 1, h_A, 0, NULL, NULL);
            //clFlush(queue);
        }
        clFinish(queue);
        t_end = magma_wtime();
        klatency = (t_end-t_start)/count*1e6;
        printf("Async Latency: %f us\n", klatency);

        clEnqueueCopyBuffer(queue, d_A, d_A, 0, 0, 1, 0, NULL, NULL);
        clFinish(queue);
        for(j=0;j<count;j++){
            if(j==0){
                t_start = magma_wtime();
            }        
            clEnqueueCopyBuffer(queue, d_A, d_B, 0, 0, 1, 0, NULL, NULL);
            //clFlush(queue);
            clFinish(queue);
        }
        t_end = magma_wtime();
        klatency = (t_end-t_start)/count*1e6;
        printf("Sync Latency: %f us\n", klatency);
        
        TESTING_FREE_DEV( d_A );
        TESTING_FREE_DEV( d_B );
    }

    if(type == 2){
        printf("testing for blas kernel launch overhead.\n");
        printf("Please comment out all the clFlush() after blas calling for accuracy.\n");
        double klatency;
        int count = 10000;
        int m = 1, n = 1, k = 1, ldda = 1, lddb = 1, lddc = 1;
        printf("k: %d\n", k);
        double alpha = 1.0, beta = 1.0;
        int transA = MagmaNoTrans, transB = MagmaNoTrans;
        
        double hA[1] = {1.0};
        
        TESTING_MALLOC_DEV( d_A, double, 1 );
        TESTING_MALLOC_DEV( d_B, double, 1 );
        TESTING_MALLOC_DEV( d_C, double, 1 );
    
        magma_dsetmatrix(1, 1, hA, 0, 1, d_A, 0, ldda, queue); 
        magma_dsetmatrix(1, 1, hA, 0, 1, d_B, 0, ldda, queue); 
        magma_dsetmatrix(1, 1, hA, 0, 1, d_C, 0, ldda, queue); 


        // DGEMM
        clAmdBlasDgemmEx(clAmdBlasColumnMajor,
                clAmdBlasNoTrans, clAmdBlasNoTrans, m, n, k, alpha, d_A, 0, ldda,
                d_B, 0, lddb, beta, d_C, 0, lddc, 
                1, &queue,
                0, NULL, NULL);
        clFinish(queue);

        int kk;
        int j;
        t_avg = 0; t_min = 1000; t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }
                clAmdBlasStatus err = clAmdBlasDgemmEx(
                    clAmdBlasColumnMajor,
                    clAmdBlasNoTrans, clAmdBlasNoTrans, m, n, k, alpha, d_A, 0, ldda,
                    d_B, 0, lddb, beta, d_C, 0, lddc, 
                    1, &queue,
                    0, NULL, NULL);
                assert(err == CL_SUCCESS);
                //clFlush(queue);
            }
            clFinish(queue);
            t_end = magma_wtime();
            magma_dprint_gpu(1, 1, d_C, 0, 1, queue);
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Dgemm Async Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Async DGEMM Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);
        
    
        t_avg = 0; t_min = 1000; t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }        
                clAmdBlasDgemmEx(clAmdBlasColumnMajor,
                    clAmdBlasNoTrans, clAmdBlasNoTrans, m, n, k, alpha, d_A, 0, ldda,
                    d_B, 0, lddb, beta, d_C, 0, lddc, 
                    1, &queue,
                    0, NULL, NULL);
                clFinish(queue);
            }
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Dgemm Sync Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Async DGEMM Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);

        // DTRMM
        magma_dtrmm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    m, n,
                    alpha, d_A, 0, ldda, 
                    d_B, 0, lddb, queue );
        clFinish(queue);

        t_avg = 0; t_min = 1000; t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }        
                magma_dtrmm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    m, n,
                    alpha, d_A, 0, ldda, 
                    d_B, 0, lddb, queue );
                //clFlush(queue);
            }
            clFinish(queue);
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Dtrmm Async Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Async DTRMM Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);

        t_avg = 0; t_min = 1000; t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }        
                magma_dtrmm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    m, n,
                    alpha, d_A, 0, ldda, 
                    d_B, 0, lddb, queue );
                clFinish(queue);
            }
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Dtrmm Sync Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Sync DTRMM Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);
        
        // Dsyrk
        magma_dsyrk(MagmaUpper, MagmaNoTrans, 
                    n, k,
                    alpha, d_A, 0, ldda,
                    beta,  d_C, 0, lddc, queue );
        clFinish(queue);

        t_avg = 0; t_min = 1000; t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }        
                magma_dsyrk(MagmaUpper, MagmaNoTrans, 
                    n, k,
                    alpha, d_A, 0, ldda,
                    beta,  d_C, 0, lddc, queue );
                //clFlush(queue);
            }
            clFinish(queue);
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Dsyrk Async Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Async DSYRK Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);
        
        t_avg = 0; t_min = 1000; t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }        
                magma_dsyrk(MagmaUpper, MagmaNoTrans, 
                    n, k,
                    alpha, d_A, 0, ldda,
                    beta,  d_C, 0, lddc, queue );
                clFinish(queue);
            }
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Dsyrk Sync Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Sync DSYRK Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);

        // Dtrsm
        magma_dtrsm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                    m, n, 
                    alpha, d_A, 0, ldda,
                    d_B, 0, lddb, queue );
        clFinish(queue);

        t_avg = 0; t_min = 1000; t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }        
                magma_dtrsm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                    m, n, 
                    alpha, d_A, 0, ldda,
                    d_B, 0, lddb, queue );
                //clFlush(queue);
            }
            clFinish(queue);
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Dtrsm Async Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Async DTRSM Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);

        t_avg = 0; t_min = 1000; t_max = 0;
        for(kk=0;kk<20;kk++){
            for(j=0;j<count;j++){
                if(j==0){
                    t_start = magma_wtime();
                }        
                magma_dtrsm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                    m, n, 
                    alpha, d_A, 0, ldda,
                    d_B, 0, lddb, queue );
                clFinish(queue);
            }
            t_end = magma_wtime();
            klatency = (t_end-t_start)/count*1e6;
            t_avg += klatency;
            if(t_min > klatency) t_min = klatency;
            if(t_max < klatency) t_max = klatency;
            printf("Dtrsm Sync Latency: %f us\n", klatency);
        }
        t_avg = t_avg/20;
        printf("Sync DTRSM Summary: avg: %f us  min: %f us  max  %f us\n", t_avg, t_min, t_max);
        
        TESTING_FREE_DEV( d_A );
        TESTING_FREE_DEV( d_B );
        TESTING_FREE_DEV( d_C );
    }

    if(type == 3){
        printf("testing for data transfer latencies.\n");
        
        cl_int err;
        double *h_A;
        size_t buffer_size = 5000*5000*sizeof(double);
        TESTING_MALLOC_DEV( d_C, double, 5000*5000 );
        d_A = clCreateBuffer(gContext, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, buffer_size, NULL, &err);
        if(CL_SUCCESS != err)
            printf("fail for d_A!\n");
        h_A = (double*)clEnqueueMapBuffer(queue, d_A, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, buffer_size, 0, NULL, NULL, &err);
        if(CL_SUCCESS != err)
            printf("fail for map h_A!\n");

        printf("buffer write.\n");
        int k;
        size_t transfer_bytes;
        double bandwidth, t_exec;
        for(k=0;k<=20;k++){
            transfer_bytes = (size_t)pow(2, k);
            int j;
            int count = 20;
            for(j=0;j<=count;j++){
                if(j==1){
                    t_start = magma_wtime();
                }
                clEnqueueWriteBuffer(queue, d_C, CL_TRUE, 0, transfer_bytes, h_A, 0, NULL, NULL);
            }
            t_end = magma_wtime();
            t_exec = (t_end-t_start)/count;
            bandwidth = transfer_bytes / t_exec /1e9;
            printf("Buffer_W: %10d\t%6.5f GB/s\t%e us\n",
                                    (int)transfer_bytes, bandwidth, t_exec*1.0e6);
        }
        
        printf("buffer read.\n");
        for(k=0;k<=20;k++){
            transfer_bytes = (size_t)pow(2, k);
            int j;
            int count = 20;
            for(j=0;j<=count;j++){
                if(j==1){
                    t_start = magma_wtime();
                }
                clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, transfer_bytes, h_A, 0, NULL, NULL);
            }
            t_end = magma_wtime();
            t_exec = (t_end-t_start)/count;
            bandwidth = transfer_bytes / t_exec /1e9;
            printf("Buffer_R: %10d\t%6.5f GB/s\t%e us\n",
                                    (int)transfer_bytes, bandwidth, t_exec*1.0e6);
        }
        if(CL_SUCCESS != clEnqueueUnmapMemObject(queue, d_A, (void*)h_A, 0, NULL, NULL))
            printf("unmapped for d_A failed!\n");
        clFinish(queue);
        clReleaseMemObject( d_A );
        //free(h_A);
        clReleaseMemObject( d_C );
    }

    magma_queue_destroy( queue );
    magma_finalize();
}
