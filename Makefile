include Makefile.inc

BUILDDIR := build

.PHONY: lib heat heat_gpu channel all clean

lib:
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/lib $(BUILDDIR)/include
	cd src; make all BUILDDIR=../$(BUILDDIR)

heat:
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	cd heat; make all BUILDDIR=../$(BUILDDIR)

heat_gpu:
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	cd heat_gpu; make all BUILDDIR=../$(BUILDDIR)

channel:
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/bin
	cd channel; make all BUILDDIR=../$(BUILDDIR)

all:
	mkdir -p $(BUILDDIR)/obj $(BUILDDIR)/lib $(BUILDDIR)/include $(BUILDDIR)/bin
	cd src; make all BUILDDIR=../$(BUILDDIR)
	cd channel; make all BUILDDIR=../$(BUILDDIR)

clean:
	cd src; make clean BUILDDIR=../$(BUILDDIR)
	cd heat; make clean BUILDDIR=../$(BUILDDIR)
	cd heat_gpu; make clean BUILDDIR=../$(BUILDDIR) || true
	cd channel; make clean BUILDDIR=../$(BUILDDIR)
	rm -rf $(BUILDDIR)
