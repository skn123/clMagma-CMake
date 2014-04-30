#------------------------------------------------------------------------------#
#The script for building testing utilities
#------------------------------------------------------------------------------#
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/testing)
#We need an auxiliary library of Fortran code
FILE (GLOB TESTING_FORTRAN_SRCS ${PROJECT_SOURCE_DIR}/testing/lin/*.f)
ADD_LIBRARY(AuxTestLib ${TESTING_FORTRAN_SRCS})
SET_TARGET_PROPERTIES(AuxTestLib PROPERTIES COMPILE_FLAGS "-x f95-cpp-input -Dmagma_devptr_t='integer(kind=8)'")

FILE (GLOB TESTING_SRCS ${PROJECT_SOURCE_DIR}/testing/*.cpp)
#From here, we remove testing_zutil.cpp
LIST( REMOVE_ITEM TESTING_SRCS 
      ${PROJECT_SOURCE_DIR}/testing/testing_util.cpp 
      ${PROJECT_SOURCE_DIR}/testing/testing_cgeqrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_cgesv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_cposv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_cpotrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_cutil.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_cgetrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dgeqrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dgesv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dposv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dpotrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dutil.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_dgetrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_sgeqrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_sgesv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_sposv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_spotrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_sutil.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_sgetrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zgeqrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zgesv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zposv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zpotrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zutil.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zgetrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zgeqrf.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zgesv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zposv.cpp
      ${PROJECT_SOURCE_DIR}/testing/testing_zpotrf.cpp

      )
      
FOREACH (SRCS ${TESTING_SRCS})
  GET_FILENAME_COMPONENT(tFile "${SRCS}" NAME_WE)
  GET_FILENAME_COMPONENT(tFile1 "${tFile}" NAME_WE)
  ADD_EXECUTABLE(${tFile1} ${SRCS})
  TARGET_LINK_LIBRARIES(${tFile1} clMagma AuxTestLib ${MODULAR_LIBRARY_LIST} ${NONMODULAR_LIBRARY_LIST})
ENDFOREACH()