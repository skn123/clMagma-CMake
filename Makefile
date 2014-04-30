#//////////////////////////////////////////////////////////////////////////////
#   -- clMAGMA (version 1.1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2014
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = .
include $(MAGMA_DIR)/Makefile.internal
-include Makefile.local

.PHONY: all lib libmagma test clean cleanall install shared

.DEFAULT_GOAL := all
all: lib test

lib: libmagma

libmagma:
	@echo ======================================== src
	( cd src              && $(MAKE) )
	@echo ======================================== control
	( cd control          && $(MAKE) )
	@echo ======================================== interface
	( cd interface_opencl && $(MAKE) )

test: lib
	@echo ======================================== testing
	( cd testing          && $(MAKE) )

clean:
	( cd control          && $(MAKE) clean )
	( cd src              && $(MAKE) clean )
	( cd interface_opencl && $(MAKE) clean )
	( cd testing          && $(MAKE) clean )
	( cd testing/lin      && $(MAKE) clean )
	-rm -f $(LIBMAGMA)

cleanall:
	( cd control          && $(MAKE) cleanall )
	( cd src              && $(MAKE) cleanall )
	( cd interface_opencl && $(MAKE) cleanall )
	( cd testing          && $(MAKE) cleanall )
	( cd testing/lin      && $(MAKE) cleanall )
	( cd lib              && rm -f *.a *.so )
	$(MAKE) cleanall2

# cleanall2 is a dummy rule to run cleanmkgen at the *end* of make cleanall, so
# .Makefile.gen files aren't deleted and immediately re-created. see Makefile.gen
cleanall2:
	@echo

dir:
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	mkdir -p $(prefix)/lib/pkgconfig

install: lib dir
	# MAGMA
	cp $(MAGMA_DIR)/include/*.h  $(prefix)/include
	cp $(LIBMAGMA)               $(prefix)/lib
	-cp $(LIBMAGMA_SO)           $(prefix)/lib
	# pkgconfig
	cat $(MAGMA_DIR)/lib/pkgconfig/clmagma.pc.in  | \
	    sed -e s:@INSTALL_PREFIX@:"$(prefix)":    | \
	    sed -e s:@INCLUDES@:"$(INC)":             | \
	    sed -e s:@LIBEXT@:"$(LIBEXT)":            | \
	    sed -e s:@MAGMA_REQUIRED@::                 \
	    > $(prefix)/lib/pkgconfig/clmagma.pc

# ========================================
# This is a crude manner of creating shared libraries.
# First create objects (with -fPIC) and static .a libraries,
# then assume all objects in these directories go into the shared libraries.
# (Except sizeptr.o and clcompile.o! That really messes things up.)
# Better solution would be to use non-recursive make, so make knows all the
# objects in each subdirectory, or use libtool, or put rules for, e.g., the
# control directory in src/Makefile (as done in src/CMakeLists.txt)
LIBMAGMA_SO = $(LIBMAGMA:.a=.so)

shared: lib
	$(MAKE) $(LIBMAGMA_SO)

$(LIBMAGMA_SO): src/*.o control/*.o interface_opencl/*.o
	@echo ======================================== $(LIBMAGMA_SO)
	rm control/sizeptr.o interface_opencl/clcompile.o
	$(CC) $(LDOPTS) -shared -o $(LIBMAGMA_SO) \
	src/*.o control/*.o \
	interface_opencl/*.o \
	$(LIBDIR) \
	$(LIB)
