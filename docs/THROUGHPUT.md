# Measured CPU throughput

The benchmark ran the pinned real GLiDE 0.4B model in CPU FP32, with four Torch threads. Existing workloads were preserved; this is not a dedicated-host or GPU benchmark. Fastino's published BF16 GPU latency is a different numerical/hardware cohort.

The encoder loaded once in **7.30 seconds**. Each native timing arm scored the same 32 old training requests, averaging 217.5 input tokens, with warmup requests excluded. Full native timings include prompt rendering, tokenization, encoder execution, frozen native readout and answer postprocessing.

| Outer request batch | Full requests/sec | Median API call |
|---:|---:|---:|
|1|1.653|606.8ms|
|2|1.719|1171.1ms for two requests|
|4|1.656|2341.2ms for four requests|
|8|1.651|4857.0ms for eight requests|

The **cached active particle-head API** measured 1.008 ms median at batch 1,1.142 ms at batch 8 and1.355 ms at batch 32. These numbers exclude the encoder and must not be presented as full-model latency.

Three separate, matched 64-update shared-PG attempts measured training scaling. They use the same normalized/residual-critic source, G=.001,D=.018,bank=.03, seed 7, horizon 64 and no EMA. Every attempt committed all 64 updates; timing excludes the first eight warm updates but those remain included in the actual update counts.

| Training batch | Updates/sec | Generator example exposures/sec |
|---:|---:|---:|
|2|58.43|116.9|
|8|32.69|261.5|
|32|12.38|396.1|

Larger batches process more examples per second but complete fewer updates per second. In the later hyperparameter search, comparing512 updates at batches8 and32 gives the larger batch four times the example exposures and approximately2.6 times the training step time. A better result there is evidence for the tested configuration; it does not isolate batch size from compute/sample budget.

The benchmark completed 192 genuine updates and exited, releasing its model before the search started. Search workers were capped at two CPU processes with four Torch threads each. Both physical GPUs were busy at inspection (99–100% utilization), and this delegated session's Babel mailbox was not prepared, so no GPU lease, allocation, launch or workload change was attempted.

Exact timings, selected request IDs, padding/token counts, source hashes, checkpoint identities and limitations are in [the machine-readable benchmark](../reports/throughput-benchmark-v1-results.json).
