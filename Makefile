all:
	nvcc -O3 -o lscl ls_cl.cu *.c
