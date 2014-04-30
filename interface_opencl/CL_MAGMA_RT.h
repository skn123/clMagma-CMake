#ifndef CL_MAGMA_RT_H
#define CL_MAGMA_RT_H
#pragma once

#include <cstdlib>
#include <vector>
#include <map>
#include <string>

#include "magma.h"

class CL_MAGMA_RT
{
    private:
        unsigned int MAX_GPU_COUNT;
        
        cl_platform_id cpPlatform;
        cl_uint ciDeviceCount;
        
        cl_kernel ckKernel;             // OpenCL kernel
        cl_event ceEvent;               // OpenCL event
        size_t szParmDataBytes;         // Byte size of context information
        size_t szKernelLength;          // Byte size of kernel code
        cl_int ciErrNum;                // Error code var
        
        bool HasBeenInitialized;
        std::map<std::string, std::string> KernelMap;
        
        int GatherFilesToCompile(const char* FileNameList, std::vector<std::string>&);
        std::string fileToString(const char* FileName);
        cl_device_id* cdDevices;        // OpenCL device list
        cl_context cxGPUContext;        // OpenCL context
        cl_command_queue *commandQueue;

        CL_MAGMA_RT();                                 // Private constructor
        ~CL_MAGMA_RT();
        
    public:

        // singleton class to guarentee only 1 instance of runtime
        static CL_MAGMA_RT * Instance()
        {
            static CL_MAGMA_RT rrt;
            return &rrt;
        }
        cl_device_id * GetDevicePtr();
        cl_context GetContext();
        cl_command_queue GetCommandQueue(int queueid);
        bool Init ();
        bool Init(cl_platform_id gPlatform, cl_context gContext);
        bool Quit ();
        
        bool CompileFile(const char*FileName, char* outDir=NULL);
        bool CompileSourceFiles(const char* FileNameList);
        const char* GetErrorCode(cl_int err);
        bool BuildFromBinaries(const char*FileName);
        bool BuildKernelMap(const char* FileNameList);
        bool CreateKernel(const char* KernelName);
        
        std::map<std::string, std::string> Kernel2FileNamePool;  // kernel name -> file name
        std::map<std::string, cl_program> ProgramPool;           // file name -> program
        std::map<std::string, cl_kernel> KernelPool;             // kernel name -> kernel
};

extern CL_MAGMA_RT *rt;

#endif        //  #ifndef CL_MAGMA_RT_H
