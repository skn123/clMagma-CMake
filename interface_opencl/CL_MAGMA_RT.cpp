#include "CL_MAGMA_RT.h"

#include <fstream>
#include <iostream>
#include <string.h>

using std::string;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::vector;


// define number of command queues to create
#define QUEUE_COUNT 1


/*
 * constructor
 */
CL_MAGMA_RT::CL_MAGMA_RT()
{
    HasBeenInitialized = false;

    cpPlatform    = NULL;
    ciDeviceCount = 0;
    cdDevices     = NULL;
    ceEvent       = NULL;
    ckKernel      = NULL;
    cxGPUContext  = NULL;
    cpPlatform    = NULL;
}

/*
 * destructor
 */
CL_MAGMA_RT::~CL_MAGMA_RT()
{
    if (!HasBeenInitialized)
        return;

    // Cleanup allocated objects
    if (commandQueue)    delete [] commandQueue;
    if (cdDevices)        free(cdDevices);
    if (ceEvent)        clReleaseEvent(ceEvent);
    if (ckKernel)    clReleaseKernel(ckKernel);
    if (cxGPUContext)    clReleaseContext(cxGPUContext);
}

cl_command_queue CL_MAGMA_RT::GetCommandQueue(int queueid)
{
    return (queueid>=QUEUE_COUNT)?NULL:commandQueue[queueid];
}

cl_device_id * CL_MAGMA_RT::GetDevicePtr()
{
    return cdDevices;
}

cl_context CL_MAGMA_RT::GetContext()
{
    return cxGPUContext;
}

/*
 * read source code from filename
 * from Rick's clutil
 */
string CL_MAGMA_RT::fileToString(const char* filename)
{
    ifstream fileStream(filename, ios::binary | ios::in | ios::ate);

    if(fileStream.is_open() == true)
    {
        size_t fileSize = fileStream.tellg();

        char* cbuffer = new char[fileSize + 1];

        fileStream.seekg(0, ios::beg);
        fileStream.read(cbuffer, fileSize);
        cbuffer[fileSize] = '\0';

        string  memoryBuffer(cbuffer);
        delete [] cbuffer;
        return memoryBuffer;
    }
    else
    {
        printf ("Error could not open %s\n", filename);
        return NULL;
    }
}


const char* CL_MAGMA_RT::GetErrorCode(cl_int err)
{
    /* TODO: this should be replaced by magma_strerror */
    switch(err)
    {
        case CL_SUCCESS:
            return "No Error.";
        case CL_INVALID_MEM_OBJECT:
            return "Invalid memory object.";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "Invalid image format descriptor.";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "Image format not supported.";
        case CL_INVALID_IMAGE_SIZE:
            return "Invalid image size.";
        case CL_INVALID_ARG_INDEX:
            return "Invalid argument index for this kernel.";
        case CL_INVALID_ARG_VALUE:
            return "Invalid argument value.";
        case CL_INVALID_SAMPLER:
            return "Invalid sampler.";
        case CL_INVALID_ARG_SIZE:
            return "Invalid argument size.";
        case CL_INVALID_BUFFER_SIZE:
            return "Invalid buffer size.";
        case CL_INVALID_HOST_PTR:
            return "Invalid host pointer.";
        case CL_INVALID_DEVICE:
            return "Invalid device.";
        case CL_INVALID_VALUE:
            return "Invalid value.";
        case CL_INVALID_CONTEXT:
            return "Invalid Context.";
        case CL_INVALID_KERNEL:
            return "Invalid kernel.";
        case CL_INVALID_PROGRAM:
            return "Invalid program object.";
        case CL_INVALID_BINARY:
            return "Invalid program binary.";
        case CL_INVALID_OPERATION:
            return "Invalid operation.";
        case CL_INVALID_BUILD_OPTIONS:
            return "Invalid build options.";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "Invalid executable.";
        case CL_INVALID_COMMAND_QUEUE:
            return "Invalid command queue.";
        case CL_INVALID_KERNEL_ARGS:
            return "Invalid kernel arguments.";
        case CL_INVALID_WORK_DIMENSION:
            return "Invalid work dimension.";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "Invalid work group size.";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "Invalid work item size.";
        case CL_INVALID_GLOBAL_OFFSET:
            return "Invalid global offset (should be NULL).";
        case CL_OUT_OF_RESOURCES:
            return "Insufficient resources.";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "Could not allocate mem object.";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "Invalid event wait list.";
        case CL_OUT_OF_HOST_MEMORY:
            return "Out of memory on host.";
        case CL_INVALID_KERNEL_NAME:
            return "Invalid kernel name.";
        case CL_INVALID_KERNEL_DEFINITION:
            return "Invalid kernel definition.";
        case CL_BUILD_PROGRAM_FAILURE:
            return "Failed to build program.";
        case CL_MAP_FAILURE:
            return "Failed to map buffer/image";
        case -1001: //This is CL_PLATFORM_NOT_FOUND_KHR
            return "No platforms found. (Did you put ICD files in /etc/OpenCL?)";
        default:
            return "Unknown error.";
    }
}

bool CL_MAGMA_RT::Quit()
{
    if (!HasBeenInitialized)
        return false;

    // Cleanup allocated objects
    if (commandQueue)    delete [] commandQueue;
    if(cdDevices)        free(cdDevices);
    if(ceEvent)        clReleaseEvent(ceEvent);
    if(ckKernel)    clReleaseKernel(ckKernel);
    if(cxGPUContext)    clReleaseContext(cxGPUContext);

    cpPlatform = NULL;
    ciDeviceCount = 0;
    cdDevices = NULL;
    ceEvent = NULL;
    ckKernel = NULL;
    cxGPUContext = NULL;
    cpPlatform = NULL;

    HasBeenInitialized = false;

    return true;
}

bool CL_MAGMA_RT::Init(cl_platform_id gPlatform, cl_context gContext)
{
  if (HasBeenInitialized)
    {
      printf ("Error: CL_MAGMA_RT has been initialized\n");
      return false;
    }

  printf ("Initializing clMAGMA runtime ...\n");
  
  cl_int ciErrNum = CL_SUCCESS;

  // set the platform
  cpPlatform    = gPlatform;

  ciErrNum  = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
  cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id));
  ciErrNum |= clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);

  // set the context
  cxGPUContext = gContext;

  // show device                                                                                                 
  for(unsigned int i = 0; i < ciDeviceCount; i++)
    {
      // get and print the device for this queue                                                                 
      //cl_device_id device = oclGetDev(cxGPUContext, i);                                                        

      char deviceName[1024], driver[1024];
      cl_ulong mem_size, alloc_size;
      memset(deviceName, '\0', 1024);
      clGetDeviceInfo(cdDevices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
      clGetDeviceInfo(cdDevices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL); 
      clGetDeviceInfo(cdDevices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &alloc_size, NULL);
      clGetDeviceInfo(cdDevices[i], CL_DRIVER_VERSION, sizeof(driver), driver, NULL);
      printf ("Device: %s (memory  %3.1f GB, max allocation  %3.1f GB, driver  %s)\n", 
               deviceName, mem_size/1.e9, alloc_size/1.e9, driver);
    }

  // create command-queues
  commandQueue = new cl_command_queue[QUEUE_COUNT];
  for(unsigned int i = 0; i < QUEUE_COUNT; i++)
    {
      // create command queue
      commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[0],
                         CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
      if (ciErrNum != CL_SUCCESS)
    {
      printf (" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
      return false;
    }
    }

  // setup kernel name -> file name (this will be done later automatically)
  // get directory from environment variable or use default
  const char* dirstr = getenv( "MAGMA_CL_DIR" );
  if ( dirstr == NULL || strlen(dirstr) == 0 ) {
        dirstr = "/usr/local/magma/cl";
        printf( "using default MAGMA_CL_DIR = %s\n", dirstr );
  }
  // make sure it ends in /
  string dir = dirstr;
  if ( dir.size() > 0 && dir[dir.size()-1] != '/' ) {
        dir += '/';
  }
  Kernel2FileNamePool["stranspose_inplace_even_kernel"] = dir + "stranspose_inplace.co";
  Kernel2FileNamePool["stranspose_inplace_odd_kernel" ] = dir + "stranspose_inplace.co";
  Kernel2FileNamePool["stranspose3_32"        ] = dir + "stranspose-v2.co";
  Kernel2FileNamePool["stranspose_32"         ] = dir + "stranspose.co";
  Kernel2FileNamePool["myslaswp2"             ] = dir + "spermute-v2.co";

  Kernel2FileNamePool["dtranspose_inplace_even_kernel"] = dir + "dtranspose_inplace.co";
  Kernel2FileNamePool["dtranspose_inplace_odd_kernel" ] = dir + "dtranspose_inplace.co";
  Kernel2FileNamePool["dtranspose3_32"        ] = dir + "dtranspose-v2.co";
  Kernel2FileNamePool["dtranspose_32"         ] = dir + "dtranspose.co";
  Kernel2FileNamePool["mydlaswp2"             ] = dir + "dpermute-v2.co";

  Kernel2FileNamePool["ctranspose_inplace_even_kernel"] = dir + "ctranspose_inplace.co";
  Kernel2FileNamePool["ctranspose_inplace_odd_kernel" ] = dir + "ctranspose_inplace.co";
  Kernel2FileNamePool["ctranspose3_32"        ] = dir + "ctranspose-v2.co";
  Kernel2FileNamePool["ctranspose_32"         ] = dir + "ctranspose.co";
  Kernel2FileNamePool["myclaswp2"             ] = dir + "cpermute-v2.co";

  Kernel2FileNamePool["ztranspose_inplace_even_kernel"] = dir + "ztranspose_inplace.co";
  Kernel2FileNamePool["ztranspose_inplace_odd_kernel" ] = dir + "ztranspose_inplace.co";
  Kernel2FileNamePool["ztranspose3_32"        ] = dir + "ztranspose-v2.co";
  Kernel2FileNamePool["ztranspose_32"         ] = dir + "ztranspose.co";
  Kernel2FileNamePool["myzlaswp2"             ] = dir + "zpermute-v2.co";

//auxiliary functions
  Kernel2FileNamePool["sset_nbxnb_to_zero"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dset_nbxnb_to_zero"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["cset_nbxnb_to_zero"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zset_nbxnb_to_zero"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset_lower"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset_lower"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset_lower"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset_lower"    ] = dir + "zauxiliary.co";
  Kernel2FileNamePool["slaset_upper"    ] = dir + "sauxiliary.co";
  Kernel2FileNamePool["dlaset_upper"    ] = dir + "dauxiliary.co";
  Kernel2FileNamePool["claset_upper"    ] = dir + "cauxiliary.co";
  Kernel2FileNamePool["zlaset_upper"    ] = dir + "zauxiliary.co";

//zlacpy functions
  Kernel2FileNamePool["slacpy_kernel"    ] = dir + "slacpy.co";
  Kernel2FileNamePool["dlacpy_kernel"    ] = dir + "dlacpy.co";
  Kernel2FileNamePool["clacpy_kernel"    ] = dir + "clacpy.co";
  Kernel2FileNamePool["zlacpy_kernel"    ] = dir + "zlacpy.co";

//zswap functions
  Kernel2FileNamePool["magmagpu_sswap"    ] = dir + "sswap.co";
  Kernel2FileNamePool["magmagpu_dswap"    ] = dir + "dswap.co";
  Kernel2FileNamePool["magmagpu_cswap"    ] = dir + "cswap.co";
  Kernel2FileNamePool["magmagpu_zswap"    ] = dir + "zswap.co";

//empty_kernel, benchmark in iwocl 2013
  Kernel2FileNamePool["sswap_empty_kernel"    ] = dir + "sswap.co";
  Kernel2FileNamePool["dswap_empty_kernel"    ] = dir + "dswap.co";
  Kernel2FileNamePool["cswap_empty_kernel"    ] = dir + "cswap.co";
  Kernel2FileNamePool["zswap_empty_kernel"    ] = dir + "zswap.co";

//dznrm2 functions
  Kernel2FileNamePool["magmablas_snrm2_adjust_kernel"    ] = dir + "snrm2.co";
  Kernel2FileNamePool["magmablas_snrm2_kernel"    ] = dir + "snrm2.co";
  Kernel2FileNamePool["magmablas_dnrm2_adjust_kernel"    ] = dir + "dnrm2.co";
  Kernel2FileNamePool["magmablas_dnrm2_kernel"    ] = dir + "dnrm2.co";
  Kernel2FileNamePool["magmablas_scnrm2_adjust_kernel"    ] = dir + "scnrm2.co";
  Kernel2FileNamePool["magmablas_scnrm2_kernel"    ] = dir + "scnrm2.co";
  Kernel2FileNamePool["magmablas_dznrm2_adjust_kernel"    ] = dir + "dznrm2.co";
  Kernel2FileNamePool["magmablas_dznrm2_kernel"    ] = dir + "dznrm2.co";

//zgemm_reduce functions
  Kernel2FileNamePool["magmablas_sgemm_reduce_kernel"    ] = dir + "sgemm_reduce.co";
  Kernel2FileNamePool["magmablas_dgemm_reduce_kernel"    ] = dir + "dgemm_reduce.co";
  Kernel2FileNamePool["magmablas_cgemm_reduce_kernel"    ] = dir + "cgemm_reduce.co";
  Kernel2FileNamePool["magmablas_zgemm_reduce_kernel"    ] = dir + "zgemm_reduce.co";

//zlarfbx functions
  Kernel2FileNamePool["magma_sgemv_kernel1"    ] = dir + "slarfbx.co";
  Kernel2FileNamePool["magma_sgemv_kernel2"    ] = dir + "slarfbx.co";
  Kernel2FileNamePool["magma_sgemv_kernel3"    ] = dir + "slarfbx.co";

  Kernel2FileNamePool["magma_dgemv_kernel1"    ] = dir + "dlarfbx.co";
  Kernel2FileNamePool["magma_dgemv_kernel2"    ] = dir + "dlarfbx.co";
  Kernel2FileNamePool["magma_dgemv_kernel3"    ] = dir + "dlarfbx.co";
  
  Kernel2FileNamePool["magma_cgemv_kernel1"    ] = dir + "clarfbx.co";
  Kernel2FileNamePool["magma_cgemv_kernel2"    ] = dir + "clarfbx.co";
  Kernel2FileNamePool["magma_cgemv_kernel3"    ] = dir + "clarfbx.co";
  
  Kernel2FileNamePool["magma_zgemv_kernel1"    ] = dir + "zlarfbx.co";
  Kernel2FileNamePool["magma_zgemv_kernel2"    ] = dir + "zlarfbx.co";
  Kernel2FileNamePool["magma_zgemv_kernel3"    ] = dir + "zlarfbx.co";

//zlarfx functions
  Kernel2FileNamePool["magma_strmv_tkernel"    ] = dir + "slarfx.co";
  Kernel2FileNamePool["magma_strmv_kernel2"    ] = dir + "slarfx.co";

  Kernel2FileNamePool["magma_dtrmv_tkernel"    ] = dir + "dlarfx.co";
  Kernel2FileNamePool["magma_dtrmv_kernel2"    ] = dir + "dlarfx.co";
  
  Kernel2FileNamePool["magma_ctrmv_tkernel"    ] = dir + "clarfx.co";
  Kernel2FileNamePool["magma_ctrmv_kernel2"    ] = dir + "clarfx.co";
  
  Kernel2FileNamePool["magma_ztrmv_tkernel"    ] = dir + "zlarfx.co";
  Kernel2FileNamePool["magma_ztrmv_kernel2"    ] = dir + "zlarfx.co";

//zlarfgx-v2 functions 
  Kernel2FileNamePool["magma_slarfgx_gpu_kernel"    ] = dir + "slarfgx-v2.co";
  Kernel2FileNamePool["magma_dlarfgx_gpu_kernel"    ] = dir + "dlarfgx-v2.co";
  Kernel2FileNamePool["magma_clarfgx_gpu_kernel"    ] = dir + "clarfgx-v2.co";
  Kernel2FileNamePool["magma_zlarfgx_gpu_kernel"    ] = dir + "zlarfgx-v2.co";

//zlag2c and clag2z
  Kernel2FileNamePool["magmaint_zlag2c"    ] = dir + "zlag2c.co";
  Kernel2FileNamePool["magmaint_dlag2s"    ] = dir + "dlag2s.co";
  Kernel2FileNamePool["clag2z_generic"    ] = dir + "clag2z.co";
  Kernel2FileNamePool["clag2z_special"    ] = dir + "clag2z.co";
  Kernel2FileNamePool["slag2d_generic"    ] = dir + "slag2d.co";
  Kernel2FileNamePool["slag2d_special"    ] = dir + "slag2d.co";


/////////////////////////////////////////////////////////////////////////////////////////
  HasBeenInitialized = true;

  BuildFromBinaries( (dir + "stranspose_inplace.co").c_str() );
  BuildFromBinaries( (dir + "stranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "stranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "spermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "dtranspose_inplace.co").c_str() );
  BuildFromBinaries( (dir + "dtranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "dtranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "dpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "ctranspose_inplace.co").c_str() );
  BuildFromBinaries( (dir + "ctranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "ctranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "cpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "ztranspose_inplace.co").c_str() );
  BuildFromBinaries( (dir + "ztranspose-v2.co"     ).c_str() );
  BuildFromBinaries( (dir + "ztranspose.co"        ).c_str() );
  BuildFromBinaries( (dir + "zpermute-v2.co"       ).c_str() );

  BuildFromBinaries( (dir + "sauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "dauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "cauxiliary.co"       ).c_str() );
  BuildFromBinaries( (dir + "zauxiliary.co"       ).c_str() );
 
  BuildFromBinaries( (dir + "slacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "dlacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "clacpy.co"       ).c_str() );
  BuildFromBinaries( (dir + "zlacpy.co"       ).c_str() );

  BuildFromBinaries( (dir + "sswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "dswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "cswap.co"       ).c_str() );
  BuildFromBinaries( (dir + "zswap.co"       ).c_str() );

  BuildFromBinaries( (dir + "snrm2.co"       ).c_str() );
  BuildFromBinaries( (dir + "dnrm2.co"       ).c_str() );
  BuildFromBinaries( (dir + "scnrm2.co"       ).c_str() );
  BuildFromBinaries( (dir + "dznrm2.co"       ).c_str() );
  
  BuildFromBinaries( (dir + "sgemm_reduce.co"       ).c_str() );
  BuildFromBinaries( (dir + "dgemm_reduce.co"       ).c_str() );
  BuildFromBinaries( (dir + "cgemm_reduce.co"       ).c_str() );
  BuildFromBinaries( (dir + "zgemm_reduce.co"       ).c_str() );
  
  BuildFromBinaries( (dir + "slarfbx.co"       ).c_str() );
  BuildFromBinaries( (dir + "dlarfbx.co"       ).c_str() );
  BuildFromBinaries( (dir + "clarfbx.co"       ).c_str() );
  BuildFromBinaries( (dir + "zlarfbx.co"       ).c_str() );
  
  BuildFromBinaries( (dir + "slarfx.co"       ).c_str() );
  BuildFromBinaries( (dir + "dlarfx.co"       ).c_str() );
  BuildFromBinaries( (dir + "clarfx.co"       ).c_str() );
  BuildFromBinaries( (dir + "zlarfx.co"       ).c_str() );

  BuildFromBinaries( (dir + "slarfgx-v2.co"       ).c_str() );
  BuildFromBinaries( (dir + "dlarfgx-v2.co"       ).c_str() );
  BuildFromBinaries( (dir + "clarfgx-v2.co"       ).c_str() );
  BuildFromBinaries( (dir + "zlarfgx-v2.co"       ).c_str() );
  
  BuildFromBinaries( (dir + "zlag2c.co"       ).c_str() );
  BuildFromBinaries( (dir + "dlag2s.co"       ).c_str() );
  BuildFromBinaries( (dir + "clag2z.co"       ).c_str() );
  BuildFromBinaries( (dir + "slag2d.co"       ).c_str() );
////////////////////////////////////////////////////////////////////////////////////  
  
  bool rtr;
  rtr = CreateKernel("stranspose_inplace_even_kernel");
  if (rtr==false)
    printf ("error creating kernel stranspose_inplace_even_kernel\n");
  rtr = CreateKernel("stranspose_inplace_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel stranspose_inplace_odd_kernel\n");
  rtr = CreateKernel("stranspose3_32");
  if (rtr==false)
    printf ("error creating kernel stranspose3_32\n");
  rtr = CreateKernel("stranspose_32");
  if (rtr==false)
    printf ("error creating kernel stranspose_32\n");
  rtr = CreateKernel("myslaswp2");
  if (rtr==false)
    printf ("error creating kernel myslaswp2\n");

  rtr = CreateKernel("dtranspose_inplace_even_kernel");
  if (rtr==false)
    printf ("error creating kernel dtranspose_inplace_even_kernel\n");
  rtr = CreateKernel("dtranspose_inplace_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel dtranspose_inplace_odd_kernel\n");
  rtr = CreateKernel("dtranspose3_32");
  if (rtr==false)
    printf ("error creating kernel dtranspose3_32\n");
  rtr = CreateKernel("dtranspose_32");
  if (rtr==false)
    printf ("error creating kernel dtranspose_32\n");
  rtr = CreateKernel("mydlaswp2");
  if (rtr==false)
    printf ("error creating kernel mydlaswp2\n");

  rtr = CreateKernel("ctranspose_inplace_even_kernel");
  if (rtr==false)
    printf ("error creating kernel ctranspose_inplace_even_kernel\n");
  rtr = CreateKernel("ctranspose_inplace_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel ctranspose_inplace_odd_kernel\n");
  rtr = CreateKernel("ctranspose3_32");
  if (rtr==false)
    printf ("error creating kernel ctranspose3_32\n");
  rtr = CreateKernel("ctranspose_32");
  if (rtr==false)
    printf ("error creating kernel ctranspose_32\n");
  rtr = CreateKernel("myclaswp2");
  if (rtr==false)
    printf ("error creating kernel myclaswp2\n");

  rtr = CreateKernel("ztranspose_inplace_even_kernel");
  if (rtr==false)
    printf ("error creating kernel ztranspose_inplace_even_kernel\n");
  rtr = CreateKernel("ztranspose_inplace_odd_kernel");
  if (rtr==false)
    printf ("error creating kernel ztranspose_inplace_odd_kernel\n");
  rtr = CreateKernel("ztranspose3_32");
  if (rtr==false)
    printf ("error creating kernel ztranspose3_32\n");
  rtr = CreateKernel("ztranspose_32");
  if (rtr==false)
    printf ("error creating kernel ztranspose_32\n");
  rtr = CreateKernel("myzlaswp2");
  if (rtr==false)
    printf ("error creating kernel myzlaswp2\n");

  rtr = CreateKernel("sset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel sset_nbxnb_zero\n");
  rtr = CreateKernel("dset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel dset_nbxnb_zero\n");
  rtr = CreateKernel("cset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel cset_nbxnb_zero\n");
  rtr = CreateKernel("zset_nbxnb_to_zero");
  if (rtr==false)
    printf ("error creating kernel zset_nbxnb_zero\n");
  rtr = CreateKernel("slaset");
  if (rtr==false)
    printf ("error creating kernel slaset\n");
  rtr = CreateKernel("dlaset");
  if (rtr==false)
    printf ("error creating kernel dlaset\n");
  rtr = CreateKernel("claset");
  if (rtr==false)
    printf ("error creating kernel claset");
  rtr = CreateKernel("zlaset");
  if (rtr==false)
    printf ("error creating kernel zlaset\n");
  rtr = CreateKernel("slaset_lower");
  if (rtr==false)
    printf ("error creating kernel slaset_lower\n");
  rtr = CreateKernel("dlaset_lower");
  if (rtr==false)
    printf ("error creating kernel dlaset_lower\n");
  rtr = CreateKernel("claset_lower");
  if (rtr==false)
    printf ("error creating kernel claset_lower");
  rtr = CreateKernel("zlaset_lower");
  if (rtr==false)
    printf ("error creating kernel zlaset_lower\n");
  rtr = CreateKernel("slaset_upper");
  if (rtr==false)
    printf ("error creating kernel slaset_upper\n");
  rtr = CreateKernel("dlaset_upper");
  if (rtr==false)
    printf ("error creating kernel dlaset_upper\n");
  rtr = CreateKernel("claset_upper");
  if (rtr==false)
    printf ("error creating kernel claset_upper");
  rtr = CreateKernel("zlaset_upper");
  if (rtr==false)
    printf ("error creating kernel zlaset_upper\n");
 
  rtr = CreateKernel("slacpy_kernel");
  if (rtr==false)
      printf ("error creating kernel slacpy_kernel\n");
  rtr = CreateKernel("dlacpy_kernel");
  if (rtr==false)
      printf ("error creating kernel dlacpy_kernel\n");
  rtr = CreateKernel("clacpy_kernel");
  if (rtr==false)
      printf ("error creating kernel clacpy_kernel");
  rtr = CreateKernel("zlacpy_kernel");
  if (rtr==false)
      printf ("error creating kernel zlacpy_kernel\n");

  rtr = CreateKernel("sswap_empty_kernel");
  if (rtr==false)
      printf ("error creating kernel sswap_empty_kernel\n");
  rtr = CreateKernel("dswap_empty_kernel");
  if (rtr==false)
      printf ("error creating kernel dswap_empty_kernel\n");
  rtr = CreateKernel("cswap_empty_kernel");
  if (rtr==false)
      printf ("error creating kernel cswap_empty_kernel\n");
  rtr = CreateKernel("zswap_empty_kernel");
  if (rtr==false)
      printf ("error creating kernel zswap_empty_kernel\n");

  rtr = CreateKernel("magmagpu_sswap");
  if (rtr==false)
      printf ("error creating kernel magmagpu_sswap\n");
  rtr = CreateKernel("magmagpu_dswap");
  if (rtr==false)
      printf ("error creating kernel magmagpu_dswap\n");
  rtr = CreateKernel("magmagpu_cswap");
  if (rtr==false)
      printf ("error creating kernel magmagpu_cswap\n");
  rtr = CreateKernel("magmagpu_zswap");
  if (rtr==false)
      printf ("error creating kernel magmagpu_zswap\n");
  
  rtr = CreateKernel("magmablas_snrm2_adjust_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_srnm2_adjust_kernel\n");
  rtr = CreateKernel("magmablas_dnrm2_adjust_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_drnm2_adjust_kernel\n");
  rtr = CreateKernel("magmablas_scnrm2_adjust_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_scrnm2_adjust_kernel\n");
  rtr = CreateKernel("magmablas_dznrm2_adjust_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_dzrnm2_adjust_kernel\n");

  rtr = CreateKernel("magmablas_snrm2_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_srnm2_kernel\n");
  rtr = CreateKernel("magmablas_dnrm2_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_drnm2_kernel\n");
  rtr = CreateKernel("magmablas_scnrm2_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_scrnm2_kernel\n");
  rtr = CreateKernel("magmablas_dznrm2_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_dzrnm2_kernel\n");

  rtr = CreateKernel("magmablas_sgemm_reduce_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_sgemm_reduce_kernel\n");
  rtr = CreateKernel("magmablas_dgemm_reduce_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_dgemm_reduce_kernel\n");
  rtr = CreateKernel("magmablas_cgemm_reduce_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_cgemm_reduce_kernel\n");
  rtr = CreateKernel("magmablas_zgemm_reduce_kernel");
  if (rtr==false)
      printf ("error creating kernel magmablas_zgemm_reduce_kernel\n");
  
  rtr = CreateKernel("magma_sgemv_kernel1");
  if (rtr==false)
      printf ("error creating kernel magma_sgemv_kernel1\n");
  rtr = CreateKernel("magma_sgemv_kernel2");
  if (rtr==false)
      printf ("error creating kernel magma_sgemv_kernel2\n");
  rtr = CreateKernel("magma_sgemv_kernel3");
  if (rtr==false)
      printf ("error creating kernel magma_sgemv_kernel3\n");
  
  rtr = CreateKernel("magma_dgemv_kernel1");
  if (rtr==false)
      printf ("error creating kernel magma_dgemv_kernel1\n");
  rtr = CreateKernel("magma_dgemv_kernel2");
  if (rtr==false)
      printf ("error creating kernel magma_dgemv_kernel2\n");
  rtr = CreateKernel("magma_dgemv_kernel3");
  if (rtr==false)
      printf ("error creating kernel magma_dgemv_kernel3\n");
  
  rtr = CreateKernel("magma_cgemv_kernel1");
  if (rtr==false)
      printf ("error creating kernel magma_cgemv_kernel1\n");
  rtr = CreateKernel("magma_cgemv_kernel2");
  if (rtr==false)
      printf ("error creating kernel magma_cgemv_kernel2\n");
  rtr = CreateKernel("magma_cgemv_kernel3");
  if (rtr==false)
      printf ("error creating kernel magma_cgemv_kernel3\n");
  
  rtr = CreateKernel("magma_zgemv_kernel1");
  if (rtr==false)
      printf ("error creating kernel magma_zgemv_kernel1\n");
  rtr = CreateKernel("magma_zgemv_kernel2");
  if (rtr==false)
      printf ("error creating kernel magma_zgemv_kernel2\n");
  rtr = CreateKernel("magma_zgemv_kernel3");
  if (rtr==false)
      printf ("error creating kernel magma_zgemv_kernel3\n");
  
  rtr = CreateKernel("magma_strmv_kernel2");
  if (rtr==false)
      printf ("error creating kernel magma_strmv_kernel2\n");
  rtr = CreateKernel("magma_strmv_tkernel");
  if (rtr==false)
      printf ("error creating kernel magma_strmv_tkernel\n");
  
  rtr = CreateKernel("magma_dtrmv_kernel2");
  if (rtr==false)
      printf ("error creating kernel magma_dtrmv_kernel2\n");
  rtr = CreateKernel("magma_dtrmv_tkernel");
  if (rtr==false)
      printf ("error creating kernel magma_dtrmv_tkernel\n");
  
  rtr = CreateKernel("magma_ctrmv_kernel2");
  if (rtr==false)
      printf ("error creating kernel magma_ctrmv_kernel2\n");
  rtr = CreateKernel("magma_ctrmv_tkernel");
  if (rtr==false)
      printf ("error creating kernel magma_ctrmv_tkernel\n");
  
  rtr = CreateKernel("magma_ztrmv_kernel2");
  if (rtr==false)
      printf ("error creating kernel magma_ztrmv_kernel2\n");
  rtr = CreateKernel("magma_ztrmv_tkernel");
  if (rtr==false)
      printf ("error creating kernel magma_ztrmv_tkernel\n");
  
  rtr = CreateKernel("magma_slarfgx_gpu_kernel");
  if (rtr==false)
      printf ("error creating kernel magma_slarfgx_gpu_kernel\n");
  rtr = CreateKernel("magma_dlarfgx_gpu_kernel");
  if (rtr==false)
      printf ("error creating kernel magma_dlarfgx_gpu_kernel\n");
  rtr = CreateKernel("magma_clarfgx_gpu_kernel");
  if (rtr==false)
      printf ("error creating kernel magma_clarfgx_gpu_kernel\n");
  rtr = CreateKernel("magma_zlarfgx_gpu_kernel");
  if (rtr==false)
      printf ("error creating kernel magma_zlarfgx_gpu_kernel\n");

  
  rtr = CreateKernel("magmaint_zlag2c");
  if (rtr==false)
      printf ("error creating kernel magmaint_zlag2c\n");
  rtr = CreateKernel("magmaint_dlag2s");
  if (rtr==false)
      printf ("error creating kernel magmaint_dlag2s\n");
  rtr = CreateKernel("clag2z_generic");
  if (rtr==false)
      printf ("error creating kernel clag2z_generic\n");
  rtr = CreateKernel("clag2z_special");
  if (rtr==false)
      printf ("error creating kernel clag2z_special\n");
  rtr = CreateKernel("slag2d_generic");
  if (rtr==false)
      printf ("error creating kernel slag2d_generic\n");
  rtr = CreateKernel("slag2d_special");
  if (rtr==false)
      printf ("error creating kernel slag2d_special\n");
  
///////////////////////////////////////////////////////////////////////////////////
  return true;
}


bool CL_MAGMA_RT::Init()
{
    if (HasBeenInitialized)
    {
        printf ("Error: CL_MAGMA_RT has been initialized\n");
        return false;
    }

    printf ("Initializing...\n");

    /*
     * initialize OpenCL runtime
     */
    cl_int ciErrNum = CL_SUCCESS;
    
    // Get the platform
    cl_uint ione = 1;
    ciErrNum = clGetPlatformIDs(1, &cpPlatform, &ione);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: Failed to create OpenCL context!\n");
        return ciErrNum;
    }

    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
    cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id));
    ciErrNum |= clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: clGetDeviceIDs at %d in file %s!\n", __LINE__, __FILE__);
        return false;
    }

    //Create the context
    cxGPUContext = clCreateContext(0, ciDeviceCount, cdDevices, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        printf("Error: Failed to create OpenCL context!\n");
        return false;
    }
        
        /*
    // Find out how many GPU's to compute on all available GPUs
    size_t nDeviceBytes;
    ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
    if (ciErrNum != CL_SUCCESS)
    {
        printf (" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
        return ciErrNum;
    }
    else if (ciDeviceCount == 0)
    {
        printf (" There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
        return false;
    }
    ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);
    */

    // show device
    for(unsigned int i = 0; i < ciDeviceCount; i++)
    {
        // get and print the device for this queue
        //cl_device_id device = oclGetDev(cxGPUContext, i);

        char deviceName[1024];
        memset(deviceName, '\0', 1024);
        clGetDeviceInfo(cdDevices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        printf ("Device: %s\n", deviceName);
    }

    // create command-queues
    commandQueue = new cl_command_queue[QUEUE_COUNT];
    for(unsigned int i = 0; i < QUEUE_COUNT; i++)
    {
        // create command queue
        commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[0], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        if (ciErrNum != CL_SUCCESS)
        {
            printf (" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
            return false;
        }
    }

    // setup kernel name -> file name (this will be done later automatically)
    string dir = "/Users/mgates/Documents/magma-cl/interface_opencl/";
    Kernel2FileNamePool["stranspose_inplace_even_kernel"] = dir + string("stranspose_inplace.co");
    Kernel2FileNamePool["stranspose_inplace_odd_kernel" ] = dir + string("stranspose_inplace.co");
    Kernel2FileNamePool["stranspose3_32"        ] = dir + string("stranspose-v2.co");
    Kernel2FileNamePool["stranspose_32"         ] = dir + string("stranspose.co");
    Kernel2FileNamePool["myslaswp2"             ] = dir + string("spermute-v2.co");

    Kernel2FileNamePool["dtranspose_inplace_even_kernel"] = dir + string("dtranspose_inplace.co");
    Kernel2FileNamePool["dtranspose_inplace_odd_kernel" ] = dir + string("dtranspose_inplace.co");
    Kernel2FileNamePool["dtranspose3_32"        ] = dir + string("dtranspose-v2.co");
    Kernel2FileNamePool["dtranspose_32"         ] = dir + string("dtranspose.co");
    Kernel2FileNamePool["mydlaswp2"             ] = dir + string("dpermute-v2.co");

    Kernel2FileNamePool["ctranspose_inplace_even_kernel"] = dir + string("ctranspose_inplace.co");
    Kernel2FileNamePool["ctranspose_inplace_odd_kernel" ] = dir + string("ctranspose_inplace.co");
    Kernel2FileNamePool["ctranspose3_32"        ] = dir + string("ctranspose-v2.co");
    Kernel2FileNamePool["ctranspose_32"         ] = dir + string("ctranspose.co");
    Kernel2FileNamePool["myclaswp2"             ] = dir + string("cpermute-v2.co");

    Kernel2FileNamePool["ztranspose_inplace_even_kernel"] = dir + string("ztranspose_inplace.co");
    Kernel2FileNamePool["ztranspose_inplace_odd_kernel" ] = dir + string("ztranspose_inplace.co");
    Kernel2FileNamePool["ztranspose3_32"        ] = dir + string("ztranspose-v2.co");
    Kernel2FileNamePool["ztranspose_32"         ] = dir + string("ztranspose.co");
    Kernel2FileNamePool["myzlaswp2"             ] = dir + string("zpermute-v2.co");

    //auxiliary functions
    Kernel2FileNamePool["sset_nbxnb_to_zero"    ] = dir + string("sauxiliary.co");
    Kernel2FileNamePool["dset_nbxnb_to_zero"    ] = dir + string("dauxiliary.co");
    Kernel2FileNamePool["cset_nbxnb_to_zero"    ] = dir + string("cauxiliary.co");
    Kernel2FileNamePool["zset_nbxnb_to_zero"    ] = dir + string("zauxiliary.co");
    Kernel2FileNamePool["slaset"    ] = dir + string("sauxiliary.co");
    Kernel2FileNamePool["dlaset"    ] = dir + string("dauxiliary.co");
    Kernel2FileNamePool["claset"    ] = dir + string("cauxiliary.co");
    Kernel2FileNamePool["zlaset"    ] = dir + string("zauxiliary.co");
    Kernel2FileNamePool["slaset_lower"    ] = dir + string("sauxiliary.co");
    Kernel2FileNamePool["dlaset_lower"    ] = dir + string("dauxiliary.co");
    Kernel2FileNamePool["claset_lower"    ] = dir + string("cauxiliary.co");
    Kernel2FileNamePool["zlaset_lower"    ] = dir + string("zauxiliary.co");
    Kernel2FileNamePool["slaset_upper"    ] = dir + string("sauxiliary.co");
    Kernel2FileNamePool["dlaset_upper"    ] = dir + string("dauxiliary.co");
    Kernel2FileNamePool["claset_upper"    ] = dir + string("cauxiliary.co");
    Kernel2FileNamePool["zlaset_upper"    ] = dir + string("zauxiliary.co");
    
    //zlacpy functions
    Kernel2FileNamePool["slacpy_kernel"    ] = dir + string("slacpy.co");
    Kernel2FileNamePool["dlacpy_kernel"    ] = dir + string("dlacpy.co");
    Kernel2FileNamePool["clacpy_kernel"    ] = dir + string("clacpy.co");
    Kernel2FileNamePool["zlacpy_kernel"    ] = dir + string("zlacpy.co");

    //zswap functions
    Kernel2FileNamePool["magmagpu_sswap"    ] = dir + string("sswap.co");
    Kernel2FileNamePool["magmagpu_dswap"    ] = dir + string("dswap.co");
    Kernel2FileNamePool["magmagpu_cswap"    ] = dir + string("cswap.co");
    Kernel2FileNamePool["magmagpu_zswap"    ] = dir + string("zswap.co");

    Kernel2FileNamePool["sswap_empty_kernel"    ] = dir + string("sswap.co");
    Kernel2FileNamePool["dswap_empty_kernel"    ] = dir + string("dswap.co");
    Kernel2FileNamePool["cswap_empty_kernel"    ] = dir + string("cswap.co");
    Kernel2FileNamePool["zswap_empty_kernel"    ] = dir + string("zswap.co");
    
    //dznrm2 functions
    Kernel2FileNamePool["magmablas_snrm2_kernel"    ] = dir + string("snrm2.co");
    Kernel2FileNamePool["magmablas_dnrm2_kernel"    ] = dir + string("dnrm2.co");
    Kernel2FileNamePool["magmablas_scnrm2_kernel"    ] = dir + string("scnrm2.co");
    Kernel2FileNamePool["magmablas_dznrm2_kernel"    ] = dir + string("dznrm2.co");

    Kernel2FileNamePool["magmablas_snrm2_adjust_kernel"    ] = dir + string("snrm2.co");
    Kernel2FileNamePool["magmablas_dnrm2_adjust_kernel"    ] = dir + string("dnrm2.co");
    Kernel2FileNamePool["magmablas_scnrm2_adjust_kernel"    ] = dir + string("scnrm2.co");
    Kernel2FileNamePool["magmablas_dznrm2_adjust_kernel"    ] = dir + string("dznrm2.co");

    //zgemm_reduce functions    
    Kernel2FileNamePool["magmablas_sgemm_reduce_kernel"    ] = dir + string("sgemm_reduce.co");
    Kernel2FileNamePool["magmablas_dgemm_reduce_kernel"    ] = dir + string("dgemm_reduce.co");
    Kernel2FileNamePool["magmablas_cgemm_reduce_kernel"    ] = dir + string("cgemm_reduce.co");
    Kernel2FileNamePool["magmablas_zgemm_reduce_kernel"    ] = dir + string("zgemm_reduce.co");
  
    //zlarfbx functions
    Kernel2FileNamePool["magma_sgemv_kernel1"    ] = dir + string("slarfbx.co");
    Kernel2FileNamePool["magma_sgemv_kernel2"    ] = dir + string("slarfbx.co");
    Kernel2FileNamePool["magma_sgemv_kernel3"    ] = dir + string("slarfbx.co");

    Kernel2FileNamePool["magma_dgemv_kernel1"    ] = dir + string("dlarfbx.co");
    Kernel2FileNamePool["magma_dgemv_kernel2"    ] = dir + string("dlarfbx.co");
    Kernel2FileNamePool["magma_dgemv_kernel3"    ] = dir + string("dlarfbx.co");
  
    Kernel2FileNamePool["magma_cgemv_kernel1"    ] = dir + string("clarfbx.co");
    Kernel2FileNamePool["magma_cgemv_kernel2"    ] = dir + string("clarfbx.co");
    Kernel2FileNamePool["magma_cgemv_kernel3"    ] = dir + string("clarfbx.co");
  
    Kernel2FileNamePool["magma_zgemv_kernel1"    ] = dir + string("zlarfbx.co");
    Kernel2FileNamePool["magma_zgemv_kernel2"    ] = dir + string("zlarfbx.co");
    Kernel2FileNamePool["magma_zgemv_kernel3"    ] = dir + string("zlarfbx.co");

    //zlarfx functions
    Kernel2FileNamePool["magma_strmv_kernel2"    ] = dir + string("slarfx.co");
    Kernel2FileNamePool["magma_strmv_tkernel"    ] = dir + string("slarfx.co");
    
    Kernel2FileNamePool["magma_dtrmv_kernel2"    ] = dir + string("dlarfx.co");
    Kernel2FileNamePool["magma_dtrmv_tkernel"    ] = dir + string("dlarfx.co");
    
    Kernel2FileNamePool["magma_ctrmv_kernel2"    ] = dir + string("clarfx.co");
    Kernel2FileNamePool["magma_ctrmv_tkernel"    ] = dir + string("clarfx.co");
    
    Kernel2FileNamePool["magma_ztrmv_kernel2"    ] = dir + string("zlarfx.co");
    Kernel2FileNamePool["magma_ztrmv_tkernel"    ] = dir + string("zlarfx.co");

    //zlag2c and clag2z
    Kernel2FileNamePool["magmaint_zlag2c"    ] = dir + string("zlag2c.co");
    Kernel2FileNamePool["magmaint_dlag2s"    ] = dir + string("dlag2s.co");
    
    Kernel2FileNamePool["clag2z_generic"    ] = dir + string("clag2z.co");
    Kernel2FileNamePool["clag2z_special"    ] = dir + string("clag2z.co");
    Kernel2FileNamePool["slag2d_generic"    ] = dir + string("slag2d.co");
    Kernel2FileNamePool["slag2d_special"    ] = dir + string("slag2d.co");

///////////////////////////////////////////////////////////////////////////////////////////
    HasBeenInitialized = true;
    return true;
}


int CL_MAGMA_RT::GatherFilesToCompile( const char* FileNameList, vector<string>& FileNames)
{
    if (FileNameList==NULL || strlen(FileNameList)==0)
        return -1;

    ifstream fileStream(FileNameList, ifstream::in);
    
    int num=0;
    if(fileStream.is_open())
    {
        while (!fileStream.eof())
        {
            char buff[512];

            fileStream.getline (buff,512);
            
            if (strlen(buff) && buff[0]!='#')
            {
                FileNames.push_back (string(buff));
                memset (buff, ' ', 512);
                num++;
            }
        }

    }
    fileStream.close();

    return num;
}

/*
 * this function build .cl files and store the bits to .o files
 */
bool CL_MAGMA_RT::CompileSourceFiles( const char* FileNameList )
{
    if (FileNameList==NULL)
        return false;

    //read from clfile for a list of cl files to compile
    vector<string> FileNames;
    int NumOfFiles = GatherFilesToCompile (FileNameList, FileNames);

    if (NumOfFiles==0)
        return false;

    //compile each cl file
    vector<string>::iterator it;
    for (it=FileNames.begin(); it<FileNames.end(); it++ )
    {
        printf ("compiling %s\n", it->c_str());
        bool ret = CompileFile (it->c_str());
        if (ret==false)
        {
            printf ("Error while trying to compile %s\n", it->c_str());
            return false;
        }
    }

    return true;
}

bool CL_MAGMA_RT::CompileFile(const char *FileName, char* outDir)
{
    if (FileName==NULL)
    {
        printf ("Error: file name empty on line %d in %s\n", __LINE__, __FILE__);
        return false;
    }

    if (!HasBeenInitialized)
        Init();

    // read in the kernel source
    string fileStrings;

    fileStrings = fileToString(FileName);
    const char *filePointers = fileStrings.c_str();

    // Create the program
    cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&filePointers, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        printf ("Error: clCreateProgramWithSource at %d in %s\n", __LINE__, __FILE__);
        return false;
    }
    
    // Build the program
    // MSUT do this otherwise clGetProgramInfo return zeros for binary sizes
    //ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-I ../include/ -I ../control/ -I /mnt/scratch/clAmdBlas-1.11.314/include/ -I/opt/AMDAPP/include/", NULL, NULL);
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-I ./", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        printf ("clBuildProgram error at %d in %s\n", __LINE__, __FILE__);
        char buildLog[16384];
        clGetProgramBuildInfo(cpProgram, cdDevices[0], CL_PROGRAM_BUILD_LOG, 
                sizeof(buildLog), buildLog, NULL);
        std::cerr << buildLog << std::endl;

        return false;
    }

    // obtain the binary
    size_t num_of_binaries=0;
    clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_of_binaries, NULL);

    size_t *binary_sizes = new size_t[num_of_binaries];

    ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, num_of_binaries*sizeof(size_t*), binary_sizes, NULL);
    if (ciErrNum!=CL_SUCCESS)
    {
        printf ("Error: clGetProgramInfo %s at line %d, file %s\n", GetErrorCode (ciErrNum), __LINE__, __FILE__);
        return false;
    }
    
    char **binaries = new char*[num_of_binaries];
    for (size_t i=0; i<num_of_binaries; i++)
        binaries[i] = new char[binary_sizes[i]];

    ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, (size_t)num_of_binaries*sizeof(unsigned char*), binaries, NULL);
    if (ciErrNum!=CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        printf ("clGetProgramInfo at %d in %s\n", __LINE__, __FILE__);
        return false;
    }

    // prepare the output file name, .cl --> .co
    string strFileName(FileName);

    //http://www.cplusplus.com/reference/cstring/strtok/
    std::vector<std::string> vString;
    char* pch = (strtok(const_cast<char*>(FileName),"/\\"));
    while(pch != NULL)
    {
      std::string v = std::string(pch);
      vString.push_back(v);
      pch = (strtok(NULL,"/\\"));
    }
    strFileName = std::string(outDir)+"/"+vString[vString.size()-1];
    vString.clear();
    
    size_t found;
    found=strFileName.find_last_of(".cl");
    strFileName.replace(found-1, 2, "co");
    
    // write binaries to files
    ofstream fileStream(strFileName.c_str(), ofstream::binary);

    if(fileStream.is_open() == true)
    {
        for (size_t i=0; i<num_of_binaries; i++)
        {
            fileStream.write ((const char *)(binary_sizes+i), (size_t)sizeof(binary_sizes[i]));
        }
        for (size_t i=0; i<num_of_binaries; i++)
            fileStream.write ((const char*)binaries[i], (size_t)binary_sizes[i]);

        fileStream.close();
    }
    else
    {
        printf ("Error: could not create binary file %s\n", strFileName.c_str());
        return false;
    }


    // cleanup
    delete [] binary_sizes;
    for (size_t i=0; i<num_of_binaries; i++)
        delete [] binaries[i];
    delete [] binaries;

    return true;
}

bool CL_MAGMA_RT::BuildFromBinaries(const char *FileName)
{
    if (FileName==NULL)
    {
        printf ("Error: file name empty on line %d in %s\n", __LINE__, __FILE__);
        return false;
    }
    
    cl_uint num_of_binaries=0;
    size_t *binary_sizes;
    unsigned char **binaries;

    // load binary from file
    ifstream fileStream(FileName, ios::binary | ios::in | ios::ate);

    if(fileStream.is_open() == true)
    {
        fileStream.seekg(0, ios::beg);

        num_of_binaries = ciDeviceCount;

        binary_sizes = new size_t[num_of_binaries];
        for (size_t i=0; i<num_of_binaries; i++)
            fileStream.read((char*)(binary_sizes+i), sizeof(binary_sizes[0]));

        binaries = new unsigned char*[num_of_binaries];
        for (size_t i=0; i<num_of_binaries; i++)
        {
            binaries[i] = new unsigned char[binary_sizes[i]];
            fileStream.read((char*)binaries[i], (size_t)binary_sizes[i]);
        }

        fileStream.close();
    }
    else
    {
        printf ("Error could not open %s\n", FileName);
        return false;
    }

    // build program from binaries
    cl_program cpProgram = clCreateProgramWithBinary(
        cxGPUContext, num_of_binaries, cdDevices,
        (const size_t*)binary_sizes, (const unsigned char **)binaries, &ciErrNum, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        printf ("clCreateProgramWithBinary failed at %d in %s\n", __LINE__, __FILE__);
        return false;
    }
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        char buildLog[16384];
        clGetProgramBuildInfo(cpProgram, cdDevices[0], CL_PROGRAM_BUILD_LOG, 
                sizeof(buildLog), buildLog, NULL);
        std::cerr << buildLog << std::endl;
        return false;
    }
    
    // put program in the pool
    ProgramPool[string(FileName)] = cpProgram;

    delete [] binary_sizes;
    for (size_t i=0; i<num_of_binaries; i++)
        delete [] binaries[i];
    delete [] binaries;

    return true;
}

/*
 * map kernel name to file
 * incomplete
 */
bool CL_MAGMA_RT::BuildKernelMap(const char *FileNameList)
{
    if (FileNameList==NULL)
        return false;

    /*
    //read from clfile for a list of cl files to compile
    vector<string> FileNames;
    int NumOfFiles = GatherFilesToCompile (FileNameList, FileNames);

    if (NumOfFiles==0)
        return false;
        */

    return true;
}

bool CL_MAGMA_RT::CreateKernel(const char *KernelName)
{
    /*
    if (!HasBeenInitialized)
    {
        printf ("Error: Uninitialized kernel\n");
        return false;
    }
    */

    cl_program cpProgram = NULL;
    //printf ("getting kernel %s from %s\n", KernelName, Kernel2FileNamePool[string(KernelName)].c_str());
    cpProgram = ProgramPool[ Kernel2FileNamePool[string(KernelName)]];
    if (cpProgram==NULL)
    {
        printf ("Error: could not find program for kernel %s\n", KernelName);
        return false;
    }

    KernelPool[string(KernelName)] = clCreateKernel(cpProgram, KernelName, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        printf ("Error: could not create kernel %s\n", KernelName);
        return false;
    }

    return true;
}
