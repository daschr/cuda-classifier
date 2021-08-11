#include "parser.h"

#include <stdlib.h>
#include <stdio.h>

bool parse_ruleset(ruleset_t *ruleset, const char *file) {
    FILE *fd=NULL;
    if((fd=fopen(file, "r"))==NULL) {
        fprintf(stderr, "ERROR: could not open file \"%s\" for ruleset!\n", file);
        return false;
    }

    if(ruleset->rules==NULL) {
        if((ruleset->rules=malloc(sizeof(rule5_t)*INITIAL_BUFSIZE))==NULL) {
            fprintf(stderr, "ERROR: could not allocate memory for rules!\n");
            goto failure;
        }

        ruleset->rules_size=INITIAL_BUFSIZE;
    }

    ruleset->num_rules=0;

#define RULES &ruleset->rules[ruleset->num_rules]
    while(fscanf(fd, "%08X %08X %08X %08X %04hX %04hX %04hX %04hX %02hhX %02hhX %02hhX\n",
                 RULES.c1[0], RULES.c1[1], RULES.c2[0], RULES.c2[1],
                 RULES.c3[0], RULES.c3[1], RULES.c4[0], RULES.c4[1],
                 RULES.c5[0], RULES.c5[1], RULES.val)==11) {
#undef RULES

        if(++(ruleset->num_rules)==ruleset->rules_size) {
            ruleset->rules_size<<=1;
            if((ruleset->rules=realloc(ruleset->rules, sizeof(rule5_t)*ruleset->rules_size))==NULL) {
                fprintf(stderr, "ERROR: could not realloc!\n");
                goto failure;
            }
        }
    }

    fclose(fd);
    return true;
failure:
    free(ruleset->rules);
    ruleset->rules=NULL;

    fclose(fd);
    return false;
}

bool parse_headers(headers_t *headers, const char *file) {
    FILE *fd=NULL;
    if((fd=fopen(file, "r"))==NULL) {
        fprintf(stderr, "ERROR: could not open file \"%s\" for headers!\n", file);
        return false;
    }

    if(headers->headers==NULL) {
        if((headers->headers=malloc(sizeof(header_t)*INITIAL_BUFSIZE))==NULL) {
            fprintf(stderr, "ERROR: could not allocate memory for headers!\n");
            goto failure;
        }

        headers->headers_size=INITIAL_BUFSIZE;
    }

    headers->num_headers=0;

#define HEADERS &headers->headers[headers->num_headers]
    while(fscanf(fd, "%08X %08X %04hX %04hX %02hhX\n",
                 HEADERS.h1, HEADERS.h2, HEADERS.h3, HEADERS.h4, HEADERS.h5)==5) {
#undef HEADERS

        if(++(headers->num_headers)==headers->headers_size) {
            headers->headers_size<<=1;
            if((headers->headers=realloc(headers->headers, sizeof(header_t)*headers->headers_size))==NULL) {
                fprintf(stderr, "ERROR: could not realloc!\n");
                goto failure;
            }
        }
    }

    fclose(fd);
    return true;
failure:
    free(headers->headers);
    headers->headers=NULL;

    fclose(fd);
    return false;
}
