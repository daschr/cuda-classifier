#ifndef _in_ls_cl
#define _in_ls_cl

#include <cuda_runtime.h>
#include "parser.h"

typedef struct{
	uint32_t *lower;
	uint32_t *upper;
	uint64_t *num_headers;
	uint32_t *header;
	uint32_t *pos;
	uint32_t *header_h;
	uint32_t *pos_h;
	unsigned char *new_pkt;
	unsigned char *done_pkt;
	unsigned char *running;
	unsigned char *new_pkt_h;
	unsigned char *done_pkt_h;
	unsigned char *running_h;
}ls_cl_t;

bool ls_cl_new(ls_cl_t *lscl, const ruleset_t *rules);
uint8_t ls_cl_get(ls_cl_t *lscl, const ruleset_t *rules, const header_t *header);
void ls_cl_free(ls_cl_t *lscl);

#endif
