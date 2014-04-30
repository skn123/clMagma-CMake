#//////////////////////////////////////////////////////////////////////////////
#   -- clMAGMA (version 1.1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2014
#//////////////////////////////////////////////////////////////////////////////

DIR = interface_opencl
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal

# alphabetic order by base name (ignoring precision)
ZSRC := \
	magmablas_z.cpp		\
	zauxiliary.cpp		\
	zbcyclic.cpp		\
	zgemm_reduce.cpp	\
	zlacpy.cpp		\
	zlag2c.cpp		\
	clag2z.cpp		\
	zlarfbx.cpp		\
	zlarfgx-v2.cpp		\
	dznrm2.cpp		\
	zpermute.cpp		\
	zswap.cpp		\
	ztranspose.cpp		\
	ztranspose2.cpp		\
	ztranspose_inplace.cpp	\

# OpenCL sources need to be here for precision generation,
# but later we separate them out into CL_SRC
ZSRC += \
	zauxiliary.cl		\
	zgemm_reduce.cl		\
	zlacpy.cl		\
	zlag2c.cl		\
	clag2z.cl		\
	zlarfbx.cl		\
	zlarfgx-v2.cl		\
	zlarfx.cl		\
	dznrm2.cl		\
	zpermute-v2.cl		\
	zswap.cl		\
	ztranspose-v2.cl	\
	ztranspose.cl		\
	ztranspose_inplace.cl	\

SRC := \
	interface.cpp	\
	strerror.cpp	\
	CL_MAGMA_RT.cpp	\

-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) $(SRC)

ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(filter-out %.cl, $(ALLOBJ)) # don't put .cl files into lib

CL_SRC := $(filter %.cl, $(ALLSRC))

CL_OBJ := $(CL_SRC:.cl=.co)

CL_COMPILER = clcompile

echo:
	@echo ALLSRC $(ALLSRC)
	@echo ALLOBJ $(ALLOBJ)
	@echo CL_SRC $(CL_SRC)
	@echo CL_OBJ $(CL_OBJ)

.PHONY: all clean cleanall

.DEFAULT_GOAL := all
all: $(LIBMAGMA) $(CL_OBJ)

$(CL_COMPILER): clcompile.o CL_MAGMA_RT.o
	$(CC) $(LDOPTS) -o $@ $^ \
	$(LIBDIR) $(LIB)

%.co: %.cl $(CL_COMPILER)
	./$(CL_COMPILER) $<

$(LIBMAGMA): $(ALLOBJ) stamp
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

# this forces LIBMAGMA to be archived
stamp: $(ALLOBJ)
	touch $@

clean:
	rm -f *.o *~ *.linkinfo stamp
	rm -f *.co $(CL_COMPILER)

cleanall: clean
	rm -f $(LIBMAGMA)
