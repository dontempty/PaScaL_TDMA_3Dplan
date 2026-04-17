include ../Makefile.inc

BUILDDIR ?= ../build

EXE    := a.out
SRCDIR := .
OBJDIR := $(BUILDDIR)/obj
IDIR   := $(BUILDDIR)/include
LDIR   := $(BUILDDIR)/lib
BINDIR := $(BUILDDIR)/bin

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJS    := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/ex_%.o, $(SOURCES))

LDFLAGS := -L$(LDIR)
LIBS    := -lpascal_tdma
CXXFLAGS += -I$(IDIR)

all: $(BINDIR)/$(EXE)

$(BINDIR)/$(EXE): $(OBJS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(OBJDIR)/ex_%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	@echo "Cleaning examples..."
	rm -f $(OBJS) $(OBJS:.o=.d) $(BINDIR)/$(EXE)

-include $(OBJS:.o=.d)
