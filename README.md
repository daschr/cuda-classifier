# cuda-classifier

* performance comparison of different kernel launch techniques (simple memcpy & launch=**simple**, zero-copy mem & async launch=**async**, zero-copy mem & persistent kernel=**persistent**)
* 5-tuple linear search for first matching rule (**highest priority**)

# branches
* simple memcpy & launch is the **master** branch
* zero-copy mem & async launch is the **async** branch
* zero-copy mem & persistent kernel is the **persistent** branch

# benchmark
* data generated using `gen_cls --size <RULES> --num_headers <HEADERS> --seed $(( RANDOM * RANDOM ))`
* machines:
  1. **NVIDIA Jetson Nano**
  2. **NVIDIA Quadro RTX 6000** and **Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz**
  
## results
|#rules|#headers|type|dur. on Jetson Nano|dur. on RTX 6000|
|------|--------|----|-------------------|----------------|
|100|100,000|**simple**|12,355,494 μs|1,671,052 μs|
|100|100,000|**async**|5,496,979 μs|301,728 μs|
|100|100,000|**persistent**|567,972 μs|1,079,522 μs|
|1,000|100,000|**simple**|12,405,534 μs|2,364,417 μs|
|1,000|100,000|**async**|10,986,671 μs|296,912 μs|
|1,000|100,000|**persistent**|770,153 μs|1,975,421 μs|
|10,000|1,000,000|**simple**|72,150,267 μs|24,375,065 μs|
|10,000|1,000,000|**async**|62,468,913 μs|4,806,557 μs|
|10,000|1,000,000|**persistent**|8,103,895 μs|27,221,085 μs|

For comparison: linear search on **Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz** with **AVX-512**
|#rules|#headers|duration|
|------|--------|--------|
|100|100,000|7651 μs|
|1,000|100,000|25792 μs|
|10,000|1,000,000|393,517 μs|
