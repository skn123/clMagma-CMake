#//////////////////////////////////////////////////////////////////////////////
#   -- clMAGMA (version 1.1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = src
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal

# ------------------------------------------------------------
# Loosely, the order is alphabetic in each section, but mixed precision,
# driver, and multi-GPU routines are separated where it made sense.
# Double precision "d" files that generate "s" files must also be listed
# in ZSRC, despite the name.

# ----------
# Cholesky, GPU interface
ZSRC := \
	zposv_gpu.cpp		\
	zpotrf_gpu.cpp		\
	zpotrf2_gpu.cpp		\
	zpotrf_mgpu.cpp		\
	zpotrf2_mgpu.cpp	\
	zpotrf_msub.cpp		\
	zpotrf2_msub.cpp	\
	zpotri_gpu.cpp		\
	zpotrs_gpu.cpp		\
	zlauum_gpu.cpp		\
	ztrtri_gpu.cpp		\

# Cholesky, CPU interface
ZSRC += \
	zposv.cpp		\
	zpotrf.cpp		\

# ----------
# LU, GPU interface
ZSRC += \
	zgesv_gpu.cpp		\
	zgetrf_gpu.cpp		\
	zgetrf2_gpu.cpp		\
	zgetrf_mgpu.cpp		\
	zgetrf2_mgpu.cpp	\
	zgetrf_msub.cpp		\
	zgetrf2_msub.cpp	\
	zgetri_gpu.cpp		\
	zgetrs_gpu.cpp		\

# LU, CPU interface
ZSRC += \
	zgesv.cpp		\
	zgetrf.cpp		\

# ----------
# QR and least squares, GPU interface
ZSRC += \
	zgels_gpu.cpp		\
	zgeqrf2_gpu.cpp		\
	zgeqrf2_2q_gpu.cpp	\
	zgeqrf_gpu.cpp		\
	zgeqrf_mgpu.cpp		\
	zgeqrf_msub.cpp		\
	zgeqrs_gpu.cpp		\
	zlarfb_gpu.cpp		\
	zunmqr_gpu.cpp		\
	zgeqr2x_gpu-v3.cpp	\

# QR, CPU interface
ZSRC += \
	zgeqrf.cpp		\
	zungqr.cpp		\
	zunmql.cpp		\
	zunmqr.cpp		\

# ----------
# symmetric eigenvalues, CPU interface
ZSRC += \
	dsyevd.cpp		\
	zheevd.cpp		\
	\
	dlaex0.cpp		\
	dlaex1.cpp		\
	dlaex3.cpp		\
	dstedx.cpp		\
	zhetrd.cpp		\
	zlatrd.cpp		\
	zstedx.cpp		\
	zunmtr.cpp		\

# ----------
# non-symmetric eigenvalues
ZSRC += \
	dgeev.cpp		\
	zgeev.cpp		\
	zgehrd.cpp		\
	zlahr2.cpp		\
	zlahru.cpp		\
	zunghr.cpp		\

# ----------
# SVD
ZSRC += \
	dgesvd.cpp		\
	zgesvd.cpp		\
	zgebrd.cpp		\
	zlabrd_gpu.cpp		\


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC_F = \
	dlaed2.f		\
	slaed2.f		\

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) $(ALLSRC_F)
ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f=.o)

.PHONY: all clean cleanall

.DEFAULT_GOAL := all
all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLHDR) $(ALLOBJ)
	touch $@

clean:
	rm -f *.o *~ *.linkinfo stamp

cleanall: clean
	rm -f $(LIBMAGMA)
