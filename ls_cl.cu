extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda_runtime.h>


#include "ls_cl.h"
#include "parser.h"
}

static inline void check_error(cudaError_t e, const char *file, int line) {
    if(e != cudaSuccess) {
        fprintf(stderr, "[ERROR] %s in %s (line %d)\n", cudaGetErrorString(e), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK(X) (check_error(X, __FILE__, __LINE__))

static inline void cpy_rules(const ruleset_t *rules, uint32_t *buffer, uint8_t upper) {
    for(size_t i=0; i<rules->num_rules; ++i) {
        buffer[i<<3]=rules->rules[i].c1[upper];
        buffer[(i<<3)+1]=rules->rules[i].c2[upper];
        buffer[(i<<3)+2]=rules->rules[i].c3[upper];
        buffer[(i<<3)+3]=rules->rules[i].c4[upper];
        buffer[(i<<3)+4]=rules->rules[i].c5[upper];
    }
}

__global__ void ls(uint *lower, uint *upper, ulong num_rules, uint *header, uint *pos) {
    uint start=(uint) blockDim.x*blockIdx.x+threadIdx.x, step=(uint) gridDim.x*blockDim.x;
	ulong bp;
    unsigned char r;
	for(uint i=start; i<num_rules; i+=step) {
        bp=i<<3;
		r= lower[bp]<=header[0] & header[0]<=upper[bp]; ++bp;
        r&=lower[bp]<=header[1] & header[1]<=upper[bp]; ++bp;
        r&=lower[bp]<=header[2] & header[2]<=upper[bp]; ++bp;
        r&=lower[bp]<=header[3] & header[3]<=upper[bp]; ++bp;
        r&=lower[bp]<=header[4] & header[4]<=upper[bp]; ++bp;
        if(r) {
            atomicMin(pos, i);
            break;
        }
    }
}

bool ls_cl_new(ls_cl_t *lscl, const ruleset_t *rules) {
    size_t bufsize=(sizeof(uint32_t)<<3)*rules->num_rules;
    uint32_t *buffer=(uint32_t *) malloc(bufsize);
    memset(buffer, 0, bufsize);
    CHECK(cudaMalloc((void **) &lscl->lower, bufsize));
    CHECK(cudaMalloc((void **) &lscl->upper, bufsize));
	
	CHECK(cudaHostAlloc((void **) &lscl->header_h, sizeof(uint32_t)<<3, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **) &lscl->pos_h, sizeof(uint64_t), cudaHostAllocMapped));

	CHECK(cudaHostGetDevicePointer((void **) &lscl->header, lscl->header_h, 0));
	CHECK(cudaHostGetDevicePointer((void **) &lscl->pos, lscl->pos_h, 0));

	cpy_rules(rules, buffer, 0);
    CHECK(cudaMemcpy(lscl->lower, buffer, bufsize, cudaMemcpyHostToDevice));

    cpy_rules(rules, buffer, 1);
    CHECK(cudaMemcpy(lscl->upper, buffer, bufsize, cudaMemcpyHostToDevice));

    free(buffer);

    return true;
}

uint8_t ls_cl_get(ls_cl_t *lscl, const ruleset_t *rules, const header_t *header) {
#define H(X) lscl->header_h[X-1]=header->h ## X
	H(1); H(2); H(3); H(4); H(5);
#undef H
	lscl->pos_h[0]=UINT_MAX;

    ls<<<512,512>>>(lscl->lower, lscl->upper, (uint64_t) rules->num_rules, lscl->header, lscl->pos);
	cudaDeviceSynchronize();

    return lscl->pos_h[0]==UINT_MAX?0xff:rules->rules[lscl->pos[0]].val;
}

void ls_cl_free(ls_cl_t *lscl) {
    cudaFree(lscl->lower);
    cudaFree(lscl->upper);
    cudaFreeHost(lscl->pos_h);
    cudaFreeHost(lscl->header_h);
}

int main(int ac, char *as[]) {
    if(ac<3) {
        fprintf(stderr, "Usage: %s [ruleset] [headers] [?result file]\n", as[0]);
        return EXIT_FAILURE;
    }

    FILE *res_file=stdout;
    if(ac>3) {
        if((res_file=fopen(as[3], "w"))==NULL) {
            fprintf(stderr, "could not open \"%s\" for writing!\n", as[3]);
            return EXIT_FAILURE;
        }
    }

    ruleset_t rules= {.num_rules=0, .rules_size=0, .rules=NULL};
    headers_t headers= {.num_headers=0, .headers_size=0, .headers=NULL};
    if(!parse_ruleset(&rules, as[1]) || !parse_headers(&headers, as[2]))
        goto fail;

    struct timeval tv1, tv2;
    ls_cl_t lscl;

    gettimeofday(&tv1, NULL);
    if(!ls_cl_new(&lscl, &rules)) {
        fputs("could not initiate ls_cl!\n", stderr);
        goto fail;
    }
    gettimeofday(&tv2, NULL);
    printf("PREPROCESSING  took %12lu us\n", 1000000*(tv2.tv_sec-tv1.tv_sec)+(tv2.tv_usec-tv1.tv_usec));

    gettimeofday(&tv1, NULL);
    for(size_t i=0; i<headers.num_headers; ++i)
        fprintf(res_file, "%02X\n", ls_cl_get(&lscl, &rules, headers.headers+i));
    gettimeofday(&tv2, NULL);
    printf("CLASSIFICATION took %12lu us\n", 1000000*(tv2.tv_sec-tv1.tv_sec)+(tv2.tv_usec-tv1.tv_usec));

    ls_cl_free(&lscl);

    return EXIT_SUCCESS;
fail:
    free(rules.rules);
    free(headers.headers);

    return EXIT_FAILURE;
}
