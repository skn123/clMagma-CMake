===================
clMAGMA README FILE (CMake Version)
===================
In order to build clMagma on Windows, you will need the following tools:
a.) CMake (http://www.cmake.org)
b.) OpenBlas (http://www.openblas.net/)
c.) clAmdBlas (http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-math-libraries/)
d.) AMD OpenCl (Sorry NVIDIA folks...This build is valid only for AMD GPU's!) (http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/)
d.) Gcc 4.8.1 (http://tdm-gcc.tdragon.net/). Install everything (including pthread, OpenMP, GFortran etc.,)
e.) MSys (containing reimp.exe) (http://www.mingw.org/wiki/MSYS)

First, download and Install MSys and set it to the path. This folder will contain
"reimp.exe" that will convert a windows library to a MinGW library (http://sourceforge.net/p/mingw/utils/ci/master/tree/reimp/)

Next, download and install OpenBlas and clAmdBlas. I use the 64-Bit versions.
Then, using reimp, convert the VisualStudio export files that these libraries
provide to GCC export files. 
****Do not modify the .dll files*****
Somewhere in your CMakeLists.txt file, add the locations of the include folders,
as well as the just-exported .a files from Both OpenBlas and clAmdBlas. Another option
would be to place them in your windows PATH and CMake will automatically find them.

Finally, download and install AMD-APP sdk. You will also need to set the path for the OpenCL folder present in AMD-APP sdk.
In this, we have libOpenCL.a (which is the exports file that will be linked to your library).

Now, follow the instructions given below for clMagma. Ignore all aspects pertaining to CUDA.
After that, continue the CMake-modified readme.

===================
clMAGMA README FILE
===================

* To INSTALL clMAGMA, modify the make.inc file to indicate where
  OpenCL BLAS, CPU BLAS, and LAPACK are installed on your system.
  Examples are given in make.inc.acml and make.inc.mkl, showing how
  to link correspondingly to ACML and MKL. After proper modification
  of the make.inc file, typing 'make', will create
   1) the libclmagma.a and libclmagmablas.a libraries in directory 'lib'
   2) testing drivers in directory 'testing'.

* To TEST clMAGMA, go to directory 'testing'. Provided are a number of
  drivers testing different routines. These drivers are also useful
  as examples on how to use clMAGMA, as well as to benchmark the performance.
  Before running set environment variable MAGMA_CL_DIR to point at
  cl_magma/interface_opencl.

* To TUNE clMAGMA, you can modify the blocking factors for the algorithms of
  interest in file 'control/get_nb_tahiti.cpp'. The default values are tuned for
  AMD Radeon 7970 (Tahiti) GPUs. You can also compare your performance to
  what we get, given in file
  'testing/results_clmagma.txt', as an easy check for your installation.

* To autotune clAcmlBlas, set the AMD_CLBLAS_STORAGE_PATH environment variable
  to a working directory and run clAmdBlasTune. Subsequent clMAGMA runs will
  use the optimized routines (as long as AMD_CLBLAS_STORAGE_PATH points
  to the directory storing the results from the clAmdBlasTune runs).

* To use a GPU on a server, disable X forwarding when you ssh to the server,
  using 'ssh -x hostname'. To see whether OpenCL finds your GPU, use 'clinfo'.

For more INFORMATION, please refer to the MAGMA homepage and user forum:

  http://icl.cs.utk.edu/magma/
  http://icl.cs.utk.edu/magma/forum/

The MAGMA project supports the package in the sense that reports of
errors or poor performance will gain immediate attention from the
developers. Such reports, descriptions of interesting applications,
and other comments should be posted on the MAGMA user forum.

===================
clMAGMA README FILE (CMake Version)
===================
To build the CMake version, and to respect the "out-of-source" build philosophy
of CMake, we had to modify a few files. Most notably, the OpenCL compiled objects (.co) files
that are generated in src/interface_opencl folder will now be generated in any user-specified folder.
In this implementation, I have set it to ${PROJECT_BINARY_DIR}. Consequently, I had to set
MAGMA_CL_DIR in the instructions provided above to ${PROJECT_BINARY_DIR}/interface_opencl. You are free to
change this to any folder of your choice. 
***However, do set the path of MAGMA_CL_DIR if you change this***.
The files that had to be changed are clCompile.cpp and CL_MAGMA_RT.cpp. 

As I had installed MinGW with all components, pthread is also installed. This means a few pre-processor definitions
had to be changed from the original clMagma source code from _WIN32/_WIN64 to _MSVC. 

==========================
Why OpenBlas and not ACML
==========================
Now, in the Original Readme, it says that this code has been tuned for AMD. This mean AMD-ACML should have been used
instead of OpenBlas. This has not been used because clMagma makes use of CBlas i.e., the C-Version of Fortran BLAS files.
Unfortunately, AMD-ACML does not provide cblas_ prefixes for these BLAS functions. Building CBlas from Source and linking it with
AMD-ACML also did not help. Hence, the only recourse was OpenBlas. The only way I foresee to obviate this problem is to rewrite 
a major chunk of clMagma code by providing #ifdef's, or any other solution. If you are able to do this for your environment, then
please share your workaround.

==========================
Testing Framework
========================== 
Most test cases will work. The one's that are omitted are listed out in the clMagmaTesting.cmake file. There are some linker errors that appear. If someone is able to fix them, please share your workaround.

==========================
More Reading
========================== 
In addition, check out the interface between Eigen and MAGMA
https://github.com/bravegag/eigen-magma

VexCl (can be interfaced with Eigen)
https://github.com/ddemidov/vexcl

and 
ViennaCl (can be interfaced with VexCl - Hence it can also interface with clMAGMA)
https://github.com/viennacl/viennacl-dev
