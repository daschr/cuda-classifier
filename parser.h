#ifndef __INCLUDE_PARSER
#define __INCLUDE_PARSER

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "rte_bv.h"

#define INITIAL_BUFSIZE 16

typedef struct {
    size_t num_rules;
    size_t rules_size;
    struct rte_table_bv_key **rules;
} ruleset_t;

bool parse_ruleset(ruleset_t *ruleset, const char *file);
void free_ruleset(ruleset_t *rules);

typedef struct {
    uint32_t h1;
    uint32_t h2;
    uint16_t h3;
    uint16_t h4;
    uint8_t h5;
} header_t;

typedef struct {
    size_t num_headers;
    size_t headers_size;
    header_t *headers;
} headers_t;

bool parse_headers(headers_t *headers, const char *file);
#endif
