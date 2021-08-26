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
    size_t bp;
    for(size_t i=0; i<rules->num_rules; ++i) {
        bp=i<<2;
        buffer[bp++]=rules->rules[i].c1[upper];
        buffer[bp++]=rules->rules[i].c2[upper];
        buffer[bp++]=(uint32_t) (rules->rules[i].c3[upper]<<16) | (uint32_t) rules->rules[i].c4[upper];
        buffer[bp]=rules->rules[i].c5[upper];
    }
}

__global__ void ls(const __restrict__ uint *lower, const __restrict__ uint *upper, const ulong rules_size,
                   const __restrict__ uint *header, uint *pos) {
    ulong i=((blockDim.x*blockIdx.x+threadIdx.x)>>2)<<2, step=(gridDim.x*blockDim.x);
    __shared__ uint8_t found;
    uint8_t r, t=threadIdx.x&3;
    const uint mask=0xf<<((threadIdx.x&31)-t);

    if(!threadIdx.x)
        found=0;

    __syncthreads();
    while(!found) {
        r=t==2?(__vcmpleu2(lower[i+t], header[2]) & __vcmpgeu2(upper[2], header[i+t]))==0xffffffff:
          lower[i+t]<=header[t] & header[t]<=upper[i+t];
        __syncwarp();
        if(__all_sync(mask, r)&(!t)) {
            atomicMin((uint *) pos, i>>2);
            found=1;
        }

        i+=step;

        if(!threadIdx.x)
            if(i>=rules_size)
                found=1;

        __syncthreads();
    }
}

bool ls_cl_new(ls_cl_t *lscl, const ruleset_t *rules) {
    size_t bufsize=(sizeof(uint32_t)<<2)*rules->num_rules;
    uint32_t *buffer=(uint32_t *) malloc(bufsize);
    memset(buffer, 0, bufsize);
    CHECK(cudaMalloc((void **) &lscl->lower, bufsize));
    CHECK(cudaMalloc((void **) &lscl->upper, bufsize));
    CHECK(cudaMalloc((void **) &lscl->header, sizeof(uint32_t)<<2));
    CHECK(cudaMalloc((void **) &lscl->pos, sizeof(uint64_t)));

    cpy_rules(rules, buffer, 0);
    CHECK(cudaMemcpy(lscl->lower, buffer, bufsize, cudaMemcpyHostToDevice));

    cpy_rules(rules, buffer, 1);
    CHECK(cudaMemcpy(lscl->upper, buffer, bufsize, cudaMemcpyHostToDevice));

    free(buffer);

    CHECK(cudaDeviceGetAttribute(&lscl->mp_count, cudaDevAttrMultiProcessorCount, 0));

    return true;
}

uint8_t ls_cl_get(ls_cl_t *lscl, const ruleset_t *rules, const header_t *header) {
#define H(X) header->h ## X
    uint32_t h[4]= { H(1), H(2), (uint32_t) (H(3)<<16) | (uint32_t) H(4), H(5) };
#undef H

    CHECK(cudaMemcpy(lscl->header, h, sizeof(uint32_t)<<2, cudaMemcpyHostToDevice));
    uint64_t p=UINT_MAX;
    CHECK(cudaMemcpy(lscl->pos, &p, sizeof(uint32_t), cudaMemcpyHostToDevice));

    ls<<<lscl->mp_count,256>>>(lscl->lower, lscl->upper, (uint64_t) rules->num_rules<<2, lscl->header, lscl->pos);

    CHECK(cudaMemcpy(&p, lscl->pos, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return p==UINT_MAX?0xff:rules->rules[p].val;
}

void ls_cl_free(ls_cl_t *lscl) {
    cudaFree(lscl->lower);
    cudaFree(lscl->upper);
    cudaFree(lscl->pos);
    cudaFree(lscl->header);
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
