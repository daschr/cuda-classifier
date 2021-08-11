#ifndef _inc_parser
#define _inc_parser

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define INITIAL_BUFSIZE 16

typedef struct {
    uint32_t c1[2];
    uint32_t c2[2];
    uint16_t c3[2];
    uint16_t c4[2];
    uint8_t c5[2];
    uint8_t val;
} rule5_t;

typedef struct {
    size_t num_rules;
    size_t rules_size;
    rule5_t *rules;
} ruleset_t;

bool parse_ruleset(ruleset_t *ruleset, const char *file);

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
