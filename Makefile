CFLAGS=-include rte_config.h -I/usr/include/dpdk -I/usr/include/libnl3 -I/usr/include/dpdk/../x86_64-linux-gnu/dpdk -I/usr/include/dpdk
CLIBS=-lrte_node -lrte_graph -lrte_bpf -lrte_flow_classify -lrte_pipeline -lrte_table -lrte_port -lrte_fib -lrte_ipsec -lrte_vhost -lrte_stack -lrte_security -lrte_sched -lrte_reorder -lrte_rib -lrte_regexdev -lrte_rawdev -lrte_pdump -lrte_power -lrte_member -lrte_lpm -lrte_latencystats -lrte_kni -lrte_jobstats -lrte_ip_frag -lrte_gso -lrte_gro -lrte_eventdev -lrte_efd -lrte_distributor -lrte_cryptodev -lrte_compressdev -lrte_cfgfile -lrte_bitratestats -lrte_bbdev -lrte_acl -lrte_timer -lrte_hash -lrte_metrics -lrte_cmdline -lrte_pci -lrte_ethdev -lrte_meter -lrte_net -lrte_mbuf -lrte_mempool -lrte_rcu -lrte_ring -lrte_eal -lrte_telemetry -lrte_kvargs -lbsd

rte_bv:
	nvcc -Werror all-warnings $(CFLAGS) -O3 -c rte_bv.c

unittest_rte_bv: rte_bv
	nvcc  -Werror all-warnings  $(CFLAGS) -o unittest_rte_bv unittest_rte_bv.c rte_bv.o $(CLIBS)

unittest_rte_table_bv: rte_bv rte_table_bv
	nvcc  -Werror all-warnings  $(CFLAGS) -o unittest_rte_table_bv unittest_rte_table_bv.c rte_table_bv.o rte_bv.o parser.c $(CLIBS)

rte_table_bv:
	nvcc -Werror all-warnings $(CFLAGS) -O3 -c rte_table_bv.cu

clean:
	rm *.o unittest_rte_bv

all:
	nvcc -O3 -o lscl ls_cl.cu *.c
