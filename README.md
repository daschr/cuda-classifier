# cuda-classifier

* performance comparison of different kernel launch techniques (simple memcpy & launch=**simple**, zero-copy mem & async launch=**async**, zero-copy mem & persistent kernel=**persistent**)
* 5-tuple linear search for first matching rule (**highest priority**)

# benchmark
* data generated using `gen_cls --size <RULES> --num_headers <HEADERS> --seed $(( RANDOM * RANDOM ))`
* machines:
  1. NVIDIA Jetson Nano
  2. NVIDIA Quadro RTX 6000 and Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz
  
## results
|#rules|#headers|type|dur. on Jetson Nano|dur. on RTX 6000|
|------|--------|----|-------------------|----------------|
|100|100,000|**simple**|12,355,494 μs|1,671,052 μs|
|100|100,000|**async**|5,496,979 μs|301,728 μs|
|100|100,000|**persistent**|567,972 μs|1,079,522 μs|
|1000|100,000|**simple**|12,405,534 μs|2,364,417 μs|
|1000|100,000|**async**|10,986,671 μs|296,912 μs|
|1000|100,000|**persistent**|770,153 μs|1,975,421 μs|
