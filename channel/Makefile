include ../Makefile.inc

# ============================================================
#  Channel Flow Solver — PaScaL_TDMA variant
#
#  Requires:  mpicxx (OpenMPI/MPICH)
#             FFTW3  at $(FFTW_DIR)
#             PaScaLTDMA lib from ../build/
#
#  Only diffusion.cpp/.hpp use PaScaL_TDMA.
#  All other sources are symlinked from ../../Filtered_TDMAv2/channel/.
# ============================================================

FFTW_DIR   := /shared/home/wel1come1234/local/fftw3
BUILDDIR   ?= ../build

EXE        := channel.out
SRCDIR     := .
OBJDIR     := $(BUILDDIR)/obj
IDIR       := $(BUILDDIR)/include
LDIR       := $(BUILDDIR)/lib
BINDIR     := $(BUILDDIR)/bin

SOURCES    := $(wildcard $(SRCDIR)/*.cpp)
OBJS       := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/ch_%.o, $(SOURCES))

# Extra includes: FFTW3 (poisson_fft.cpp uses it) and PaScaLTDMA
CXXFLAGS   += -I$(FFTW_DIR)/include -I$(IDIR)

# Link: PaScaLTDMA lib + FFTW3 (still needed for poisson_fft)
LDFLAGS    := -L$(LDIR) -L$(FFTW_DIR)/lib
LIBS       := -lpascal_tdma -lfftw3_mpi -lfftw3 -lm

# Rpath so the binary finds FFTW at runtime
LDFLAGS    += -Wl,-rpath,$(FFTW_DIR)/lib

all: dirs $(BINDIR)/$(EXE)

dirs:
	mkdir -p $(OBJDIR) $(BINDIR)

$(BINDIR)/$(EXE): $(OBJS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(OBJDIR)/ch_%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(BINDIR)/$(EXE)
