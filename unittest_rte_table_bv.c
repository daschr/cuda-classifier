#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>

#include <rte_eal.h>
#include "rte_table_bv.h"
#include "parser.h"

int main(int ac, char *as[]){
	if(ac==1){
			fprintf(stderr, "Usage: %s [rules]\n", as[0]);
			return EXIT_FAILURE;
	}

	if(rte_eal_init(ac, as)<0)
			rte_exit(EXIT_FAILURE, "Error: could not init EAL.\n");
	
	ruleset_t ruleset;
	memset(&ruleset, 0, sizeof(ruleset_t));

	if(!parse_ruleset(&ruleset, as[1])){
			fprintf(stderr, "Error: could not parse ruleset \"%s\"\n", as[1]);
			rte_eal_cleanup();
			return EXIT_FAILURE;
	}

	struct rte_table_bv_field_def fdefs[4];
	for(size_t i=0;i<4;++i){
		fdefs[i].offset=8;
		fdefs[i].type=RTE_TABLE_BV_FIELD_TYPE_RANGE;
		fdefs[i].size=32>>(i>>1);
	}

	struct rte_table_bv_params table_params = { .num_fields=4, .field_defs=fdefs };

	puts("creating table");	
	void *table=rte_table_bv_ops.f_create(&table_params, rte_socket_id(), 0);	
	printf("table: %p\n", table);

	puts("freeing table");
	rte_table_bv_ops.f_free(table);	

	puts("rte_eal_cleanup()");
	rte_eal_cleanup();
	return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
