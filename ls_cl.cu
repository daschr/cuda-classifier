extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "ls_cl.h"
#include "parser.h"
}

#define RINGBUF_SIZE 32
#define RINGBUF_MASK 31

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

void *get_results(void *p) {
    ls_cl_t *lscl=(ls_cl_t *) p;
    bool stream_running;
    int32_t pos;
    for(size_t i=0;; i=(i+1)&RINGBUF_MASK) {
        do {
            pthread_mutex_lock(&lscl->running_mtxs[i]);
            stream_running=lscl->streams_running[i];
            pthread_mutex_unlock(&lscl->running_mtxs[i]);
            if(!lscl->running&&!stream_running) goto end;
        } while(!stream_running);

        cudaStreamSynchronize(lscl->streams[i]);
        pos=*lscl->pos_ring_h[i];

        fprintf(lscl->outfile, "%02X\n", pos==UINT_MAX?0xff:lscl->ruleset->rules[pos].val);
        *lscl->pos_ring_h[i]=UINT_MAX;

        pthread_mutex_lock(&lscl->running_mtxs[i]);
        lscl->streams_running[i]=0;
        pthread_mutex_unlock(&lscl->running_mtxs[i]);
    }
end:
    return NULL;
}

__global__ void ls(	const __restrict__ uint *lower, const __restrict__ uint *upper, const ulong rules_size,
                    const __restrict__ uint *header, uint *pos) {

    ulong start=blockDim.x*blockIdx.x+threadIdx.x, step=(gridDim.x*blockDim.x)<<2;
    __shared__ uint8_t found;
    __shared__ uint h[4];
    ulong i;
    uint8_t r;

    if(!threadIdx.x) {
        found=0;
#pragma unroll
        for(int i=0; i<4; ++i)
            h[i]=header[i];
        __threadfence_block();
    }

    __syncthreads();
    i=start<<2;
    while(!found) {
        r=i<rules_size?lower[i]<=h[0] & h[0]<=upper[i]
          & lower[i+1]<=h[1] & h[1]<=upper[i+1]
          & (__vcmpleu2(lower[i+2], h[2]) & __vcmpgeu2(upper[i+2], h[2]))==0xffffffff
          & lower[i+3]<=h[3] & h[3]<=upper[i+3]:0;

        if(r) {
            atomicMin((uint *) pos, i>>2);
            found=1;
            __threadfence_system();
        }

        if((!start) & (i>rules_size))
            found=1;

        i+=step;
        __syncthreads();
    }
}

bool ls_cl_new(ls_cl_t *lscl, const ruleset_t *rules, FILE *outfile) {
    lscl->ruleset=rules;
    lscl->streams_running=(uint8_t *) malloc(sizeof(uint8_t)*RINGBUF_SIZE);
    memset(lscl->streams_running, 0, sizeof(uint8_t)*RINGBUF_SIZE);
    lscl->running=1;
    lscl->outfile=outfile;

    lscl->running_mtxs=(pthread_mutex_t *) malloc(sizeof(pthread_mutex_t)*RINGBUF_SIZE);
    for(size_t i=0; i<RINGBUF_SIZE; ++i)
        lscl->running_mtxs[i]=PTHREAD_MUTEX_INITIALIZER;

    // lower upper buffer

    size_t bufsize=(sizeof(uint32_t)<<2)*rules->num_rules;
    uint32_t *buffer=(uint32_t *) malloc(bufsize);
    memset(buffer, 0, bufsize);

    CHECK(cudaMalloc((void **) &lscl->lower, bufsize));
    CHECK(cudaMalloc((void **) &lscl->upper, bufsize));

    cpy_rules(rules, buffer, 0);
    CHECK(cudaMemcpy(lscl->lower, buffer, bufsize, cudaMemcpyHostToDevice));

    cpy_rules(rules, buffer, 1);
    CHECK(cudaMemcpy(lscl->upper, buffer, bufsize, cudaMemcpyHostToDevice));

    // head pos ring buffer
    lscl->pos_ring_h=(uint32_t **) malloc(sizeof(uint32_t *)*RINGBUF_SIZE);
    lscl->pos_ring=(uint32_t **) malloc(sizeof(uint32_t *)*RINGBUF_SIZE);
    lscl->header_ring_h=(uint32_t **) malloc(sizeof(uint32_t *)*RINGBUF_SIZE);
    lscl->header_ring=(uint32_t **) malloc(sizeof(uint32_t *)*RINGBUF_SIZE);


    for(size_t i=0; i<RINGBUF_SIZE; ++i) {
        CHECK(cudaHostAlloc((void **) &(lscl->header_ring_h[i]), (sizeof(uint32_t)<<2), cudaHostAllocMapped));
        CHECK(cudaHostGetDevicePointer((void **) &(lscl->header_ring[i]), lscl->header_ring_h[i], 0));
        CHECK(cudaHostAlloc((void **) &(lscl->pos_ring_h[i]), sizeof(uint32_t), cudaHostAllocMapped));
        CHECK(cudaHostGetDevicePointer((void **) &(lscl->pos_ring[i]), lscl->pos_ring_h[i], 0));
    }

    lscl->streams=(cudaStream_t *) malloc(sizeof(cudaStream_t)*RINGBUF_SIZE);
    for(size_t i=0; i<RINGBUF_SIZE; ++i)
        CHECK(cudaStreamCreateWithFlags(lscl->streams+i, 0));

    CHECK(cudaDeviceGetAttribute(&lscl->mp_count, cudaDevAttrMultiProcessorCount, 0));

    pthread_create(&lscl->getrest, NULL, get_results, (void *) lscl);

    free(buffer);

    return true;
}

void ls_cl_get(ls_cl_t *lscl, const header_t *header) {
    static uint32_t i=0;

    lscl->header_ring_h[i][0]=header->h1;
    lscl->header_ring_h[i][1]=header->h2;
    lscl->header_ring_h[i][2]=((uint32_t) header->h3<<16)|(uint32_t) header->h4;
    lscl->header_ring_h[i][3]=header->h5;

    ls<<<1,128,0,lscl->streams[i]>>>(lscl->lower, lscl->upper, (uint64_t) lscl->ruleset->num_rules<<2,
                                     lscl->header_ring[i], lscl->pos_ring[i]);

    uint8_t stream_running;
    do {
        pthread_mutex_lock(&lscl->running_mtxs[i]);
        stream_running=lscl->streams_running[i];
        pthread_mutex_unlock(&lscl->running_mtxs[i]);
    } while(stream_running);


    pthread_mutex_lock(&lscl->running_mtxs[i]);
    lscl->streams_running[i]=1;
    pthread_mutex_unlock(&lscl->running_mtxs[i]);

    i=(i+1)&RINGBUF_MASK;
}

void ls_cl_free(ls_cl_t *lscl) {
    //lscl->running=0;
    //pthread_join(lscl->getrest, NULL);
    cudaFree(lscl->lower);
    cudaFree(lscl->upper);
    for(size_t i=0; i<RINGBUF_SIZE; ++i) {
        cudaFreeHost(lscl->pos_ring_h[i]);
        cudaFreeHost(lscl->header_ring_h[i]);
    }
    free(lscl->pos_ring);
    free(lscl->header_ring);
    free(lscl->pos_ring_h);
    free(lscl->header_ring_h);
    free(lscl->streams_running);
    free(lscl->streams);
    free(lscl->running_mtxs);
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
    if(!ls_cl_new(&lscl, &rules, res_file)) {
        fputs("could not initiate ls_cl!\n", stderr);
        goto fail;
    }
    gettimeofday(&tv2, NULL);
    printf("PREPROCESSING  took %12lu us\n", 1000000*(tv2.tv_sec-tv1.tv_sec)+(tv2.tv_usec-tv1.tv_usec));

    gettimeofday(&tv1, NULL);
    for(size_t i=0; i<headers.num_headers; ++i)
        ls_cl_get(&lscl, headers.headers+i);
    lscl.running=0;
    pthread_join(lscl.getrest, NULL);
    gettimeofday(&tv2, NULL);
    printf("CLASSIFICATION took %12lu us\n", 1000000*(tv2.tv_sec-tv1.tv_sec)+(tv2.tv_usec-tv1.tv_usec));

    ls_cl_free(&lscl);

    return EXIT_SUCCESS;
fail:
    free(rules.rules);
    free(headers.headers);

    return EXIT_FAILURE;
}
