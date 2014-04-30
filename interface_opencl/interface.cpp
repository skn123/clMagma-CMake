/*
 *   -- clMAGMA (version 1.1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2014
 *
 * @author Mark Gates
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "magma.h"
#include "CL_MAGMA_RT.h"

#ifdef HAVE_clAmdBlas

// ========================================
// globals
cl_platform_id gPlatform;
cl_context     gContext;

magma_event_t  *gevent;


// Run time global variable used for LU
CL_MAGMA_RT *rt;

// ========================================
// initialization
magma_err_t
magma_init()
{
    cl_int err;
    err = clGetPlatformIDs( 1, &gPlatform, NULL );
    assert( err == 0 );
    
    cl_device_id devices[ MagmaMaxGPUs ];
    cl_uint num;
    err = clGetDeviceIDs( gPlatform, CL_DEVICE_TYPE_GPU, MagmaMaxGPUs, devices, &num );
    assert( err == 0 );
    
    cl_context_properties properties[3] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties) gPlatform, 0 };
    gContext = clCreateContext( properties, num, devices, NULL, NULL, &err );
    assert( err == 0 );
    
    err = clAmdBlasSetup();
    assert( err == 0 );

    // Initialize kernels related to LU
    rt = CL_MAGMA_RT::Instance();
    rt->Init(gPlatform, gContext);
    
    gevent = NULL;

    return err;
}

// --------------------
magma_err_t
magma_finalize()
{
    cl_int err;
    clAmdBlasTeardown();
    err = clReleaseContext( gContext );

    // quit the RT
    rt->Quit();

    return err;
}

// --------------------
// Print the available GPU devices. Used in testing.
void magma_print_devices()
{
    cl_uint ndevices;

    clGetDeviceIDs(gPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ndevices);
    cl_device_id* Devices = (cl_device_id *)malloc(ndevices * sizeof(cl_device_id));
    clGetDeviceIDs(gPlatform, CL_DEVICE_TYPE_GPU, ndevices, Devices, NULL);

    for(unsigned int i = 0; i < ndevices; i++)
      {
        char deviceName[1024], driver[1024];
        cl_ulong mem_size, alloc_size;
        memset(deviceName, '\0', 1024);

        clGetDeviceInfo(Devices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        clGetDeviceInfo(Devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
        clGetDeviceInfo(Devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &alloc_size, NULL);
        clGetDeviceInfo(Devices[i], CL_DRIVER_VERSION, sizeof(driver), driver, NULL);
        printf ("Device: %s (memory  %3.1f GB, max allocation  %3.1f GB, driver  %s)\n",
                deviceName, mem_size/1.e9, alloc_size/1.e9, driver);
      }
    free( Devices );
}


// ========================================
// memory allocation
magma_err_t
magma_malloc( magma_ptr* ptrPtr, size_t size )
{
    cl_int err;
    *ptrPtr = clCreateBuffer( gContext, CL_MEM_READ_WRITE, size, NULL, &err );
    return err;
}

// --------------------
magma_err_t
magma_free( magma_ptr ptr )
{
    cl_int err = clReleaseMemObject( ptr );
    return err;
}

// --------------------
// Allocate size bytes on CPU, returning pointer in ptrPtr.
// The purpose of using this instead of malloc() is to properly align arrays
// for vector (SSE) instructions. The default implementation uses
// posix_memalign (on Linux, MacOS, etc.) or _aligned_malloc (on Windows)
// to align memory to a 32 byte boundary.
// Use magma_free_cpu() to free this memory.
extern "C"
magma_err_t magma_malloc_cpu( void** ptrPtr, size_t size )
{
#if 1
    #if defined( _WIN32 ) || defined( _WIN64 )
    *ptrPtr = _aligned_malloc( size, 32 );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    #else
    int err = posix_memalign( ptrPtr, 32, size );
    if ( err != 0 ) {
        *ptrPtr = NULL;
        return MAGMA_ERR_HOST_ALLOC;
    }
    #endif
#else
    *ptrPtr = malloc( size );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
    return MAGMA_SUCCESS;
}

// --------------------
// Free CPU pinned memory previously allocated by magma_malloc_pinned.
// The default implementation uses free(), which works for both malloc and posix_memalign.
// For Windows, _aligned_free() is used.
extern "C"
magma_err_t magma_free_cpu( void* ptr )
{
#if defined( _WIN32 ) || defined( _WIN64 )
    _aligned_free( ptr );
#else
    free( ptr );
#endif
    return MAGMA_SUCCESS;
}


// ========================================
// device & queue support
magma_err_t
magma_get_devices(
    magma_device_t* devices,
    magma_int_t     size,
    magma_int_t*    numPtr )
{
    cl_int err;
    //err = clGetDeviceIDs( gPlatform, CL_DEVICE_TYPE_GPU, 1, size, devices, num );
    size_t n;
    err = clGetContextInfo(
        gContext, CL_CONTEXT_DEVICES,
        size*sizeof(magma_device_t), devices, &n );
    *numPtr = n / sizeof(magma_device_t);
    return err;
}

// --------------------
magma_int_t 
magma_num_gpus( void )
{
    const char *ngpu_str = getenv("MAGMA_NUM_GPUS");
    cl_uint ngpu = 1;
    if ( ngpu_str != NULL ) {
        char* endptr;
        ngpu = strtol( ngpu_str, &endptr, 10 );

        cl_uint ndevices;    
        clGetDeviceIDs(gPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ndevices);

        if ( ngpu < 1 || *endptr != '\0' ) {
          ngpu = 1;
          fprintf( stderr, "$MAGMA_NUM_GPUS=%s is an invalid number; using %d GPU.\n",
                   ngpu_str, ngpu );
        }
        else if ( ngpu > MagmaMaxGPUs || ngpu > ndevices ) {
          ngpu = ((ndevices < MagmaMaxGPUs)? ndevices : MagmaMaxGPUs);
          fprintf( stderr, "$MAGMA_NUM_GPUS=%s exceeds MagmaMaxGPUs=%d or available GPUs=%d; using %d GPUs.\n",
                   ngpu_str, MagmaMaxGPUs, ndevices, ngpu );
        }
        assert( 1 <= ngpu && ngpu <= ndevices );
    }
    return (magma_int_t)ngpu;
}

// --------------------                                                                                                      
magma_int_t 
magma_queue_meminfo( magma_queue_t queue )
{
    cl_device_id dev;
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &dev, NULL);
  
    cl_ulong mem_size;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);   
    mem_size /= sizeof(magmaDoubleComplex); 

    return mem_size;
}

// --------------------
magma_err_t
magma_queue_create( magma_device_t device, magma_queue_t* queuePtr )
{
    assert( queuePtr != NULL );
    cl_int err;
    #ifdef TRACING
    *queuePtr = clCreateCommandQueue( gContext, device, CL_QUEUE_PROFILING_ENABLE, &err );
    #else
    *queuePtr = clCreateCommandQueue( gContext, device, 0, &err );
    #endif
    return err;
}

// --------------------
magma_err_t
magma_queue_destroy( magma_queue_t  queue )
{
    cl_int err = clReleaseCommandQueue( queue );
    return err;
}

// --------------------
magma_err_t
magma_queue_sync( magma_queue_t queue )
{
    cl_int err = clFinish( queue );
    clFlush( queue );
    return err;
}


// ========================================
// event support
magma_err_t
magma_setevent( magma_event_t* event )
{
  #ifdef TRACING
    gevent = event;
  #else
    printf("%s not implemented\n", __func__ );
  #endif

  return 0;
}

magma_err_t
magma_event_create( magma_event_t* event )
{
    printf( "%s not implemented\n", __func__ );
    return 0;
}

magma_err_t
magma_event_destroy( magma_event_t event )
{
    printf( "%s not implemented\n", __func__ );
    return 0;
}

magma_err_t
magma_event_record( magma_event_t event, magma_queue_t queue )
{
    printf( "%s not implemented\n", __func__ );
    return 0;
}

magma_err_t
magma_event_query( magma_event_t event )
{
    printf( "%s not implemented\n", __func__ );
    return 0;
}

magma_err_t
magma_event_sync( magma_event_t event )
{
    cl_int err = clWaitForEvents(1, &event);
    return err;
}

#endif // HAVE_clAmdBlas
