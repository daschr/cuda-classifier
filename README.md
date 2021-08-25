# cuda-classifier

* performance comparison of different kernel launch techniques for packet classification (simple memcpy & launch=**simple**, zero-copy mem & async launch=**async**, zero-copy mem & persistent kernel=**persistent**)
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
  3. **NVIDIA GTX 1650**
* NOTES
  - **async** uses ring buffers (maybe i'll try the same for **persistent**)
  - DMA is faster due to SoC
  
|#rules|#headers|type|dur. on GTX 1650|dur. on RTX 6000|dur. on Jetson Nano|
|------|--------|----|--------|--------|-----------|
|100|100.000|**simple**|1.870.565 μs|1.631.440 μs|12.818.421 μs|
|100|100.000|**async**|174.572 μs|287.919 μs|3.374.396 μs|
|100|100.000|**persistent**|1.147.622 μs|677.997 μs|501.097 μs|
|1.000|100.000|**simple**|1.542.390 μs|1.812.467 μs|12.626.683 μs|
|1.000|100.000|**async**|176.059 μs|292.031 μs|9.873.650 μs|
|1.000|100.000|**persistent**|1.116.731 μs|606.281 μs|556.832 μs|
|10.000|100.000|**simple**|1.534.786 μs|1.624.867 μs|12.442.075 μs|
|10.000|100.000|**async**|174.678 μs|262.784 μs|4.209.705 μs|
|10.000|100.000|**persistent**|1.179.979 μs|648.732 μs|613.345 μs|
|100|1.000.000|**simple**|14.078.265 μs|14.528.700 μs|123.435.719 μs|
|100|1.000.000|**async**|1.803.910 μs|2.787.523 μs|39.789.084 μs|
|100|1.000.000|**persistent**|11.128.599 μs|6.247.920 μs|5.032.594 μs|
|1.000|1.000.000|**simple**|15.240.088 μs|18.517.724 μs|123.740.656 μs|
|1.000|1.000.000|**async**|1.793.599 μs|2.731.569 μs|68.457.302 μs|
|1.000|1.000.000|**persistent**|11.567.677 μs|6.306.792 μs|5.539.173 μs|
|10.000|1.000.000|**simple**|15.285.675 μs|19.055.859 μs|124.249.198 μs|
|10.000|1.000.000|**async**|1.742.507 μs|2.699.068 μs|61.602.149 μs|
|10.000|1.000.000|**persistent**|11.761.187 μs|6.381.792 μs|6.224.600 μs|

For comparison: linear search on **Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz** with **AVX-512**
|#rules|#headers|duration|
|------|--------|--------|
|100|100,000|7.651 μs|
|1,000|100,000|25.792 μs|
|10,000|1,000,000|393,517 μs|
