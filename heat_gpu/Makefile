include ../Makefile.inc

BUILDDIR ?= ../build

EXE    := heat_gpu.out
SRCDIR := .
OBJDIR := $(BUILDDIR)/obj
IDIR   := $(BUILDDIR)/include
LDIR   := $(BUILDDIR)/lib
BINDIR := $(BUILDDIR)/bin

CPP_SOURCES := $(wildcard $(SRCDIR)/*.cpp)
CU_SOURCES  := $(wildcard $(SRCDIR)/*.cu)
CPP_OBJS    := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/exgpu_%.o, $(CPP_SOURCES))
CU_OBJS     := $(patsubst $(SRCDIR)/%.cu,  $(OBJDIR)/exgpu_%.o, $(CU_SOURCES))
OBJS        := $(CPP_OBJS) $(CU_OBJS)

LDFLAGS := -L$(LDIR)
# pascal_tdma static archive contains CUDA objects, so we need the CUDA runtime.
CUDA_LIBDIR ?= $(if $(CUDA_HOME),$(CUDA_HOME)/lib64,/usr/local/cuda/lib64)
# Derive include dir from libdir (works for the KISTI nvhpc/25.11 layout).
CUDA_INCDIR ?= $(CUDA_LIBDIR:/lib64=/include)
LIBS    := -lpascal_tdma -L$(CUDA_LIBDIR) $(CUDA_LDFLAGS) -lstdc++
CXXFLAGS += -I$(IDIR) -I$(CUDA_INCDIR)

all: $(BINDIR)/$(EXE)

# Re-link when libpascal_tdma.a is refreshed (its content depends on src/*.cu/cpp).
$(BINDIR)/$(EXE): $(OBJS) $(LDIR)/libpascal_tdma.a | $(BINDIR)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR)/exgpu_%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

$(OBJDIR)/exgpu_%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -I$(IDIR) -c $< -o $@ -MMD
-include $(CU_OBJS:.o=.d)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	@echo "Cleaning heat_gpu..."
	rm -f $(OBJS) $(CPP_OBJS:.o=.d) $(CU_OBJS:.o=.d) $(BINDIR)/$(EXE)

-include $(CPP_OBJS:.o=.d)
