#//////////////////////////////////////////////////////////////////////////////
#   -- clMAGMA (version 1.1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = testing
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal

# ----------
# BLAS and auxiliary
ZSRC := \
	testing_zgemm.cpp	\
	testing_zgemm_reduce.cpp	\
	testing_zgemv.cpp	\
	\
	testing_constants.cpp	\
	testing_zlaswp.cpp	\

# ----------
# Cholesky, GPU interface
ZSRC += \
	testing_zposv_gpu.cpp	\
	testing_zpotrf_gpu.cpp	\
	testing_zpotrf2_gpu.cpp	\
	testing_zpotrf_msub.cpp	\
	testing_zpotri_gpu.cpp	\
	\
	testing_zpotrf_mgpu.cpp	\

# Cholesky, CPU interface
ZSRC += \
	testing_zposv.cpp	\
	testing_zpotrf.cpp	\

# ----------
# LU, GPU interface
ZSRC += \
	testing_zgesv_gpu.cpp	\
	testing_zgetrf_gpu.cpp	\
	testing_zgetrf_msub.cpp	\
	testing_zgetrf2_gpu.cpp	\
	testing_zgetri_gpu.cpp	\
	\
	testing_zgetrf_mgpu.cpp	\

# LU, CPU interface
ZSRC += \
	testing_zgesv.cpp	\
	testing_zgetrf.cpp	\

# ----------
# QR and least squares, GPU interface
ZSRC += \
	testing_zgeqrf_gpu.cpp	\
	testing_zgeqrf_msub.cpp	\
	testing_zgeqrf2_gpu.cpp	\
	testing_zgeqrs_gpu.cpp	\
	testing_zgeqr2x_gpu.cpp \
	\
	testing_zgeqrf_mgpu.cpp	\

# QR, CPU interface
ZSRC += \
	testing_zgeqrf.cpp	\

# ----------
# symmetric eigenvalues, CPU interface
ZSRC += \
	testing_dsyevd.cpp	\
	testing_zheevd.cpp	\
	testing_zhetrd.cpp	\

# ----------
# non-symmetric eigenvalues
ZSRC += \
	testing_dgeev.cpp	\
	testing_zgeev.cpp	\
	testing_zgehrd.cpp	\

# ----------
# SVD
ZSRC += \
	testing_zgesvd.cpp	\
	testing_zgebrd.cpp	\

# ----------
# iwocl benchmark
ZSRC += \
	testing_benchmark.cpp	\


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f90=.o)
ALLOBJ := $(ALLOBJ:.F90=.o)
ALLOBJ := $(ALLOBJ:.cuf=.o)

EXE    := $(ALLOBJ:.o=)

# --------------------
LIBTEST := libtest.a
LIBSRC  := testing_zutil.o testing_cutil.o testing_dutil.o testing_sutil.o testing_util.o
LIBOBJ  := $(LIBSRC:.cpp=.o)
LIBEXE  := $(LIBOBJ:.o=)

# testing_zutil.cpp isn't in ZSRC (since they don't generate EXEs),
# so add generation rules here instead of .Makefile.gen
testing_cutil.cpp: testing_zutil.cpp
	$(PYTHON) $(MAGMA_DIR)/tools/codegen.py  -p c --file testing_zutil.cpp

testing_dutil.cpp: testing_zutil.cpp
	$(PYTHON) $(MAGMA_DIR)/tools/codegen.py  -p d --file testing_zutil.cpp

testing_sutil.cpp: testing_zutil.cpp
	$(PYTHON) $(MAGMA_DIR)/tools/codegen.py  -p s --file testing_zutil.cpp

gen += testing_cutil.cpp testing_dutil.cpp  testing_sutil.cpp

generate: testing_cutil.cpp testing_dutil.cpp testing_sutil.cpp


# --------------------
.PHONY: all lib clean cleanall cleanexe

.DEFAULT_GOAL := all
all: $(EXE)

lib:
	cd .. && $(MAKE) lib

# depend on header
$(ALLOBJ): testings.h

$(LIBTEST): $(LIBOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@

LIBLAPACKTEST = lin/liblapacktest.a

$(LIBLAPACKTEST):
	cd lin && $(MAKE) liblapacktest.a

$(EXE): $(LIBMAGMA) $(LIBLAPACKTEST) $(LIBTEST)

clean:
	cd lin    && $(MAKE) clean
	rm -f *.o *~ $(LIBTEST)

cleanexe:
	rm -f $(EXE)

cleanall: clean cleanexe

testing_% : testing_%.o
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	$(LIBTEST) $(LIBLAPACKTEST) -L$(MAGMA_DIR)/lib -lclmagma \
	$(LIBDIR) \
	$(LIB)

# keep intermediate object files, to speed up re-compiling when library changes
.SECONDARY: $(ALLOBJ)
