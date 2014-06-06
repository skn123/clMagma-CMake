#ifndef clVendor_H_
#define clVendor_H_
// http://stackoverflow.com/questions/7001424/opencl-problem-with-double-type
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#endif