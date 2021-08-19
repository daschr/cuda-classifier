#ifndef _in_ls_cl
#define _in_ls_cl

#include <cuda_runtime.h>
#include <pthread.h>
#include "parser.h"
#include "stdbool.h"

typedef struct{
	bool running;
	pthread_t getrest;
	int mp_count;
	uint32_t *lower;
	uint32_t *upper;
	uint32_t **header_ring;
	uint32_t **pos_ring;
	uint32_t **header_ring_h;
	uint32_t **pos_ring_h;
	const ruleset_t *ruleset;
	pthread_mutex_t *running_mtxs;
	uint8_t *streams_running;
	cudaStream_t *streams;
	FILE *outfile;
}ls_cl_t;

bool ls_cl_new(ls_cl_t *lscl, const ruleset_t *rules, FILE *outfile);
void ls_cl_get(ls_cl_t *lscl, const header_t *header);
void ls_cl_free(ls_cl_t *lscl);

#endif
