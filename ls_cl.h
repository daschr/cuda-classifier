#ifndef _in_ls_cl
#define _in_ls_cl

#include <cuda_runtime.h>
#include "parser.h"

typedef struct{
	int mp_count;
	uint32_t *lower;
	uint32_t *upper;
	uint64_t *num_headers;
	uint32_t *header;
	uint32_t *pos;
}ls_cl_t;

bool ls_cl_new(ls_cl_t *lscl, const ruleset_t *rules);
uint8_t ls_cl_get(ls_cl_t *lscl, const ruleset_t *rules, const header_t *header);
void ls_cl_free(ls_cl_t *lscl);

#endif
