CFLAGS=$(shell pkg-config --cflags libdpdk)
CLIBS=$(shell pkg-config --libs libdpdk)

rte_bv:
	nvcc -Werror all-warnings $(CFLAGS) -O3 -c rte_bv.c

unittest_rte_bv: rte_bv
	nvcc  -Werror all-warnings  $(CFLAGS) -o unittest_rte_bv unittest_rte_bv.c rte_bv.o $(CLIBS)

rte_table_bv:
	nvcc -Werror all-warnings $(CFLAGS) -O3 -c rte_table_bv.cu

clean:
	rm *.o unittest_rte_bv

all:
	nvcc -O3 -o lscl ls_cl.cu *.c
