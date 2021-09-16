#ifndef __INCLUDE_RTE_TABLE_BV__
#define __INCLUDE_RTE_TABLE_BV__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <dpdk/rte_table.h>

#define RTE_TABLE_BV_MAX_RANGES 0xffffff
#define RTE_TABLE_BV_BS	(RTE_TABLE_BV_MAX_RANGES>>5)

enum {
	RTE_TABLE_BV_FIELD_TYPE_RANGE,
	RTE_TABLE_BV_FIELD_TYPE_BITMASK
};

struct rte_table_bv_field_def {
	uint32_t offset; // offset from data start
	
	uint8_t type;
	uint8_t size; // in bytes
};

struct rte_table_bv_field {
	uint32_t value;
	uint32_t mask_range;
};

struct rte_table_bv_params {
	uint32_t num_rules;

	uint32_t num_fields;
	// size needs to be  >=num_fields
	const struct rte_table_bv_field_def *field_defs;
};

struct rte_table_bv_key {
	uint32_t *buf; // size = sum(field_defs[*].size)
	uint32_t val;
};

#ifdef __cplusplus
}
#endif

#endif
