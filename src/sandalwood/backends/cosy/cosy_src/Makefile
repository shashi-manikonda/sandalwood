# Makefile for COSY INFINITY on Linux and macOS
#
# Definition of FC, FFLAGS, LINK and LIBS depends on each local system.
# We appreciate receiving information on your local makefile.
#
# Using gfortran of GNU GCC
# Uncomment the next gfortran lines, and comment out the ifort lines
#
FC     = gfortran
FFLAGS = -std=legacy -ffixed-form -O1
LINK   = gfortran

#
# Using ifort of Intel Fortran
#  -fp-model strict is needed for verified numerics
#
#FC     = ifort
#FFLAGS = -fp-model strict -ip
#LINK   = ifort

#
# If compiling with PGPLOT, uncomment the next PGPLOT LIBS line
#
LIBS   = 
#LIBS   = -L/usr/local/pgplot -lpgplot -L/usr/X11R6/lib -lX11

OBJ    = foxy.o dafox.o foxfit.o foxgraf.o

cosy: $(OBJ)
	$(LINK) -o cosy $(OBJ) $(LIBS)
