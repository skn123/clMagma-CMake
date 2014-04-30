#------------------------------------------------------------------------------#
#CMakeLists for clMagma 1.1.0
#Prerequisites: clAmdBlas, OpenBlas, OpenCL
#------------------------------------------------------------------------------#
PROJECT(clMagma CXX Fortran)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
#This Line pertains to my dev environment. If you want to follow your own 
#mechanism, then you will probably need
#FIND_PACKAGE(OpenCL)
#FIND_PACKAGE(OpenBlas)
#FIND_PACKAGE(clAmdBlas)

FIND_PACKAGE(ThirdPartyLibraries_MinGW)
SET(clcompile_SRCS
    ${PROJECT_SOURCE_DIR}/interface_opencl/clcompile.cpp
    ${PROJECT_SOURCE_DIR}/interface_opencl/CL_MAGMA_RT.cpp
)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include
					          ${PROJECT_SOURCE_DIR}/control
                    ${PROJECT_SOURCE_DIR}/interface_opencl)

ADD_DEFINITIONS(-DHAVE_clAmdBlas -DHAVE_CBLAS -DADD_)
ADD_EXECUTABLE(clCompile ${clcompile_SRCS})
TARGET_LINK_LIBRARIES(clCompile ${MODULAR_LIBRARY_LIST})
SET_TARGET_PROPERTIES(clCompile PROPERTIES 
                      COMPILE_FLAGS "-DHAVE_clAmdBlas -DHAVE_CBLAS -DADD_")
                      
#------------------------------------------------------------------------------#
#This variable enumerates all the cpp/c and header files that will be used in 
#building the library
#------------------------------------------------------------------------------#
FILE(GLOB clMagma_SRCS 
  ${PROJECT_SOURCE_DIR}/src/*.cpp
  ${PROJECT_SOURCE_DIR}/src/*.c
  ${PROJECT_SOURCE_DIR}/src/*.h 
  ${PROJECT_SOURCE_DIR}/interface_opencl/*.cpp
  ${PROJECT_SOURCE_DIR}/interface_opencl/*.c
  ${PROJECT_SOURCE_DIR}/interface_opencl/*.h 
  ${PROJECT_SOURCE_DIR}/include/*.h
  ${PROJECT_SOURCE_DIR}/control/*.h
  ${PROJECT_SOURCE_DIR}/control/*.cpp
  ${PROJECT_SOURCE_DIR}/control/*.c
)
#Remove a couple of files that were part of clCompile
LIST( REMOVE_ITEM clMagma_SRCS 
      ${PROJECT_SOURCE_DIR}/interface_opencl/clcompile.cpp 
      ${PROJECT_SOURCE_DIR}/interface_opencl/magmablas_z_trace.cpp 
      ${PROJECT_SOURCE_DIR}/interface_opencl/magmablas_z_trace2.cpp 
      ${PROJECT_SOURCE_DIR}/control/sizeptr.c
      ${PROJECT_SOURCE_DIR}/src/zpotrf2_mgpu_trace.cpp
      ${PROJECT_SOURCE_DIR}/src/zpotrf_gpu_trace.cpp
)

#------------------------------------------------------------------------------#
#This variable enumerates all the fortran files that will be used in 
#building the library
#------------------------------------------------------------------------------#
FILE(GLOB_RECURSE clMagma_Fortran_SRCS 
     ${PROJECT_SOURCE_DIR}/src/*.f*
     ${PROJECT_SOURCE_DIR}/control/*.f*
     ${PROJECT_SOURCE_DIR}/interface_opencl/*.f*)

#------------------------------------------------------------------------------#
#The OpenCL files will be integrated into the library differently. First we
#accumulate all the files 
#------------------------------------------------------------------------------#
FILE(GLOB clMagma_CL_SRCS 
  ${PROJECT_SOURCE_DIR}/interface_opencl/*.cl
)
SET(clMagma_CL_OBJS)
SET(TGT_DIR ${CMAKE_CURRENT_BINARY_DIR}/interface_opencl)
#http://stackoverflow.com/questions/3702115/creating-a-directory-in-cmake
FILE(MAKE_DIRECTORY ${TGT_DIR})

#------------------------------------------------------------------------------#
#The file "kernels_header.h" present in /interface_opencl folder needs to be
#copied to the binary dir where clCompile is built. A better solution would be
#to include the path of the file in the program itself (TBD)
#------------------------------------------------------------------------------#
EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy 
                 ${PROJECT_SOURCE_DIR}/interface_opencl/kernels_header.h
                 ${CMAKE_CURRENT_BINARY_DIR}
)

#------------------------------------------------------------------------------#
#Build the Fortran files into a separate archive
#TODO: Using ADD_LIBRARY(clMagma_Fortran OBJECT...) make CMake complain
#"CMake Error: Target clMagma_Fortran has dependency information when it shouldn't.
#Your cache is probably stale. Please remove the entry
#  clMagma_Fortran_LIB_DEPENDS
#from the cache.
#------------------------------------------------------------------------------#
ADD_LIBRARY(clMagma_Fortran ${clMagma_Fortran_SRCS})
SET_TARGET_PROPERTIES(clMagma_Fortran PROPERTIES COMPILE_FLAGS "-x f95-cpp-input -Dmagma_devptr_t='integer(kind=8)'")

#------------------------------------------------------------------------------#
#Build the clMagma Library
#------------------------------------------------------------------------------#
ADD_LIBRARY(clMagma SHARED ${clMagma_SRCS})
TARGET_LINK_LIBRARIES(clMagma clMagma_Fortran ${MODULAR_LIBRARY_LIST} ${NONMODULAR_LIBRARY_LIST})
SET_TARGET_PROPERTIES(clMagma PROPERTIES 
                      COMPILE_FLAGS "-DADD_ -DHAVE_clAmdBlas -DHAVE_CBLAS"
                      PREFIX "")
#------------------------------------------------------------------------------#
#Follow the instructions given on this webpage
#http://www.cmake.org/pipermail/cmake/2008-November/025541.html 
#------------------------------------------------------------------------------#
FOREACH(clFile ${clMagma_CL_SRCS})
  GET_FILENAME_COMPONENT(tFile "${clFile}" NAME_WE)
  SET(oFile2 ${CMAKE_CURRENT_BINARY_DIR}/interface_openCL/${tFile}.co)
  LIST(APPEND clMagma_CL_OBJS ${oFile2})
  ADD_CUSTOM_COMMAND( TARGET clMagma
    COMMAND clCompile ${clFile} ${TGT_DIR}
  )
ENDFOREACH()

SET(Build_Testing OFF CACHE BOOL "Build clMagma Testing")
IF(Build_Testing)
INCLUDE(${PROJECT_SOURCE_DIR}/clMagmaTesting.cmake)
ENDIF()