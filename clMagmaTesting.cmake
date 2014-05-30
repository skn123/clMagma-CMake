#------------------------------------------------------------------------------#
#The script for building testing utilities
#------------------------------------------------------------------------------#
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/testing)
#We need an auxiliary library of Fortran code
FILE (GLOB TESTING_FORTRAN_SRCS ${PROJECT_SOURCE_DIR}/testing/lin/*.f)
ADD_LIBRARY(AuxTestLib ${TESTING_FORTRAN_SRCS})
SET_TARGET_PROPERTIES(AuxTestLib PROPERTIES COMPILE_FLAGS "-x f95-cpp-input -Dmagma_devptr_t='integer(kind=8)'")
#We also need an auxiliary library of util code
ADD_LIBRARY(AuxTestLib2 OBJECT 
           ${PROJECT_SOURCE_DIR}/testing/testing_util.cpp 
           ${PROJECT_SOURCE_DIR}/testing/testing_cutil.cpp
           ${PROJECT_SOURCE_DIR}/testing/testing_dutil.cpp
           ${PROJECT_SOURCE_DIR}/testing/testing_sutil.cpp
           ${PROJECT_SOURCE_DIR}/testing/testing_zutil.cpp
           )
FILE (GLOB TESTING_SRCS ${PROJECT_SOURCE_DIR}/testing/*.cpp)
#From here, we remove testing_zutil.cpp
LIST( REMOVE_ITEM TESTING_SRCS 
      #These files do not have a main
      ${PROJECT_SOURCE_DIR}/testing/testing_util.cpp 
      ${PROJECT_SOURCE_DIR}/testing/testing_cutil.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dutil.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_sutil.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zutil.cpp
      #These files produce an error when linking - slamch_ / dlamch_ also defined in OpenBlas
      ${PROJECT_SOURCE_DIR}/testing/testing_cheevd.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dsyevd.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_ssyevd.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zheevd.cpp
      #These files do not need testing_utils
      ${PROJECT_SOURCE_DIR}/testing/testing_cgeqr2x_gpu.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dgeqr2x_gpu.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_sgeqr2x_gpu.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zgeqr2x_gpu.cpp
      )
      
FOREACH (SRCS ${TESTING_SRCS})
  GET_FILENAME_COMPONENT(tFile "${SRCS}" NAME_WE)
  GET_FILENAME_COMPONENT(tFile1 "${tFile}" NAME_WE)
  ADD_EXECUTABLE(${tFile1} ${SRCS} $<TARGET_OBJECTS:AuxTestLib2> )
  TARGET_LINK_LIBRARIES(${tFile1} clMagma AuxTestLib ${MODULAR_LIBRARY_LIST} ${NONMODULAR_LIBRARY_LIST})
ENDFOREACH()

#These tests are done differently as there is a redefinition of a function in each cpp file
#as well as a in testing_utils.cpp
SET(TESTING_SRCS1 ${PROJECT_SOURCE_DIR}/testing/testing_cgeqr2x_gpu.cpp
          ${PROJECT_SOURCE_DIR}/testing/testing_dgeqr2x_gpu.cpp
          ${PROJECT_SOURCE_DIR}/testing/testing_sgeqr2x_gpu.cpp
          ${PROJECT_SOURCE_DIR}/testing/testing_zgeqr2x_gpu.cpp
)
FOREACH (SRCS ${TESTING_SRCS1})
  GET_FILENAME_COMPONENT(tFile "${SRCS}" NAME_WE)
  GET_FILENAME_COMPONENT(tFile1 "${tFile}" NAME_WE)
  ADD_EXECUTABLE(${tFile1} ${SRCS})
  TARGET_LINK_LIBRARIES(${tFile1} clMagma AuxTestLib ${MODULAR_LIBRARY_LIST} ${NONMODULAR_LIBRARY_LIST})
ENDFOREACH()

