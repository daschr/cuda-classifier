#include "parser.h"

#include "rte_table_bv.h"
#include <stdlib.h>
#include <stdio.h>

bool parse_ruleset(ruleset_t *ruleset, const char *file) {
    FILE *fd=NULL;
    if((fd=fopen(file, "r"))==NULL) {
        fprintf(stderr, "ERROR: could not open file \"%s\" for ruleset!\n", file);
        return false;
    }

    if(ruleset->rules==NULL) {
        if((ruleset->rules=malloc(sizeof(struct rte_table_bv_key *)*INITIAL_BUFSIZE))==NULL) {
            fprintf(stderr, "ERROR: could not allocate memory for rules!\n");
            goto failure;
        }

        for(size_t i=0; i<INITIAL_BUFSIZE; ++i) {
            if((ruleset->rules[i]=malloc(sizeof(struct rte_table_bv_key)))==NULL) {
                fprintf(stderr, "ERROR: could not allocate memory for %luth rule!\n", i);
                goto failure;
            }

            if((ruleset->rules[i]->buf=malloc(sizeof(uint32_t)*10))==NULL) {
                fprintf(stderr, "ERROR: could not  allocate memory for %luth rule!\n", i);
                goto failure;
            }
        }

        ruleset->rules_size=INITIAL_BUFSIZE;
    }

    ruleset->num_rules=0;

#define RULES &ruleset->rules[ruleset->num_rules]->buf
    while(fscanf(fd, "%08X %08X %08X %08X %04X %04X %04X %04X %02X %02X %02X\n",
                 RULES[0], RULES[1], RULES[2], RULES[3],
                 RULES[4], RULES[5], RULES[6], RULES[7],
                 RULES[8], RULES[9], &ruleset->rules[ruleset->num_rules]->val)==11) {
#undef RULES

        ruleset->rules[ruleset->num_rules]->pos=ruleset->num_rules;

        if(++(ruleset->num_rules)==ruleset->rules_size) {
            ruleset->rules_size<<=1;
            if((ruleset->rules=realloc(ruleset->rules, sizeof(struct rte_table_bv_key *)*ruleset->rules_size))==NULL) {
                fprintf(stderr, "ERROR: could not realloc!\n");
                goto failure;
            }

            for(size_t i=ruleset->rules_size>>1; i<ruleset->rules_size; ++i) {
                if((ruleset->rules[i]=malloc(sizeof(struct rte_table_bv_key)))==NULL) {
                    fprintf(stderr, "ERROR: could not allocate memory for %luth rule!\n", i);
                    goto failure;
                }

                if((ruleset->rules[i]->buf=malloc(sizeof(uint32_t)*10))==NULL) {
                    fprintf(stderr, "ERORR: could not allocate memory for %luth rule!\n", i);
                    goto failure;
                }
            }
        }
    }

    fclose(fd);
    return true;

failure:
    if(ruleset->rules) {
        for(size_t i=0; i<ruleset->rules_size; ++i) {
            free(ruleset->rules[i]->buf);
            free(ruleset->rules[i]);
        }

        free(ruleset->rules);
    }

    ruleset->rules=NULL;

    fclose(fd);
    return false;
}

void free_ruleset(ruleset_t *ruleset) {
    if(ruleset->rules) {
        for(size_t i=0; i<ruleset->rules_size; ++i) {
            free(ruleset->rules[i]->buf);
            free(ruleset->rules[i]);
        }

        free(ruleset->rules);
    }
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
