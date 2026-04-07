include Makefile.inc

BUILDDIR := build

lib:
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/lib $(BUILDDIR)/include
	cd src; make all BUILDDIR=../$(BUILDDIR)

example:
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	cd examples; make all BUILDDIR=../$(BUILDDIR)

all:
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/lib $(BUILDDIR)/include $(BUILDDIR)/bin
	cd src; make all BUILDDIR=../$(BUILDDIR)
	cd examples; make all BUILDDIR=../$(BUILDDIR)

clean:
	cd src; make clean BUILDDIR=../$(BUILDDIR)
	cd examples; make clean BUILDDIR=../$(BUILDDIR)
	rm -rf $(BUILDDIR)
