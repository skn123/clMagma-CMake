#include "CL_MAGMA_RT.h"

#include "mkl.h"

#include "NB.h"


extern "C" void magmablas_sinplace_transpose_cpu(float *A, int lda, int n );
extern "C" void magma_sinplace_transpose( cl_mem A, int offset, int lda, int m, magma_queue_t queue );
double get_cur_time();


/*
 * verify_answer
 * 1. pull data from device to host
 * 2. verify
 */
extern "C" void verify_answer (int m, int n, float *refC, float *oclC, float& resid)
{
    int i;
    
    for (i=0; i<n*m; i++)
    {
        //printf ("oclC(%d)=%f\trefC(%d)=%f\n", i+1, oclC[i], i+1, refC[i]);
        oclC[i] -= refC[i];
    }
    
    resid = (float)slange_ ("F", &m, &n, oclC, &m, NULL)/(float)slange_ ("F", &m, &n, refC, &m, NULL);
}

void test_transpose(int size, int offset, CL_MAGMA_RT *rt)
{
    int M, N, K;
    float *A, *B;
    cl_mem cmB;        // OpenCL buffers for M, V, and W
    unsigned int mem_size_A, mem_size_B;
    cl_int ciErrNum;                // Error code var
    
    if (offset%128!=0)
    {
        printf ("Error: offset must be a multiple of %d\n", NB);
        return;
    }
    
    M=N=K=size;
    printf ("--------------------------------------------------\n");
    printf ("testing size %d, %d/%d=%d\n", size, size, 32, size/32);
    
    //--------- Allocate and initialize host arrays -----------//
    {
        printf ("allocate host arrays..."); fflush (stdout);
        mem_size_A = M * K * sizeof(float);
        A = (float*)malloc(mem_size_A);
        
        mem_size_B = K * N * sizeof(float);
        B = (float*)malloc(mem_size_B);
        
        if (!A || !B)
        {
            printf ("Error: could not allocating host matrices\n");
            return;
        }
        
        int j;
        for (j = 0; j < M*K; j++)
            A[j] = rand() / (float)RAND_MAX;
        
        memcpy (B, A, M*K*sizeof(float));
    }
    
    //----------- Allocate the OpenCL buffer memory objects ----------------//
    {
        printf ("allocate gpu arrays of size %d, %p...", mem_size_B, rt); fflush (stdout);
        cmB = clCreateBuffer(rt->GetContext(), CL_MEM_READ_WRITE, mem_size_B, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS)
        {
            printf("Error: clCreateBuffer at %d in file %s!\n", __LINE__, __FILE__);
            return;
        }
        
        printf ("done\n");
    }
    
    //-------------- write data to GPU device ----------------//
    {
        printf ("copy data to GPU..."); fflush (stdout);
        ciErrNum = clEnqueueWriteBuffer(rt->GetCommandQueue(0), cmB, CL_TRUE, 0, mem_size_B, B, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS)
        {
            printf("Error: clEnqueueWriteBuffer at %d in file %s!\n", __LINE__, __FILE__);
            return;
        }
        printf ("done\n");
    }
    
    int lda=M, ldb=K;
    
    offset = 128;
    int offsetrange = offset*offset;
    //----------- call cpu sinplace ------------//
    {
        printf ("cpu transpose..."); fflush (stdout);
        magmablas_sinplace_transpose_cpu(A+offsetrange, lda, M-offset);
    }
    
    //----------- call opencl sgemm -------------//
    printf ("gpu transpose..."); fflush (stdout);
    magma_sinplace_transpose(cmB, offsetrange, ldb, K-offset, rt->GetCommandQueue(0) );
    
    //----------- check result ---------------//
    // Read back results and check accumulated errors
    ciErrNum = clEnqueueReadBuffer(rt->GetCommandQueue(0), cmB, CL_TRUE, 0, mem_size_B, (void*)B, 0, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clEnqueueWriteBuffer at %d in file %s!\n", __LINE__, __FILE__);
        return;
    }
    
    float resid;
    verify_answer (M, N, A, B, resid);
    
    printf ("resid=%e\n", resid);
    
    // clean up
    if (cmB)clReleaseMemObject(cmB);
    free(A);
    free(B);
}

#define TEST_COMPILE
//#define TEST_LOADING

int main()
{
    CL_MAGMA_RT runtime;
    
    runtime.Init();
    
#ifdef TEST_COMPILE
    runtime.CompileSourceFiles("clfiles");
    runtime.Quit();
    runtime.Init();
#endif
    
    runtime.BuildFromBinaries("sinplace_transpose.co");
    runtime.BuildFromBinaries("stranspose-v2.co");
    runtime.BuildFromBinaries("stranspose.co");
    runtime.BuildFromBinaries("spermute-v2.co");
    
    bool ret;
    ret = runtime.CreateKernel("sinplace_T_even_kernel");
    if (ret==false)
        printf ("error creating kernel sinplace_T_even_kernel\n");
    ret = runtime.CreateKernel("sinplace_T_odd_kernel");
    if (ret==false)
        printf ("error creating kernel sinplace_T_odd_kernel\n");
        
    // remove this when finishing debugging
    ret = runtime.CreateKernel("myslaswp2");
    if (ret==false)
        printf ("error creating kernel sinplace_T_odd_kernel\n");
    /////////////////////////////////////////
    
    // test even size
    int M=1024;
    int offset = 0;
    test_transpose (M, offset, &runtime);
    
    // test odd size
    M=1024+32;
    test_transpose (M, offset, &runtime);
    return 0;
}
