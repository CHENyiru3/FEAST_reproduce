# Study 07 CUDA canary status

Updated: 2026-07-20  
Status: pending external GPU execution allowance

The installed FEAST candidate passed focused CUDA log-domain parity tests, and
exact audits completed both the former E18.5 failure block and an E15.5
24,994,980-pair block on the RTX A6000. No Study 07 workflow shard has yet been
created.

Further elevated Python/CUDA launches were rejected because the execution
service exhausted its current usage allowance. This is not a FEAST exception,
CUDA out-of-memory event, or method failure. The sandbox cannot see
`/dev/nvidia*`, and the workflow intentionally does not fall back to CPU.

When GPU execution is available, run these bounded gates in order from this
directory with
`/maiziezhou_lab2/yiru/envs/feast-publication-cuda-py311/bin/python`:

```bash
python run.py --age E15.5 --shard-id canary_otlog_v5_e15_z12 --z-indices 12
python run.py --age E18.5 --shard-id canary_otlog_v5_e18_edge_middle --z-indices 0 100
```

Validate their per-slice records, hashes, exact identity, and positive solver
evidence before expanding to the complete declared canary sets:

```bash
python run.py --age E15.5 --shard-id canary_otlog_v5_e15_remaining --z-indices 0 78 47 157
python run.py --age E18.5 --shard-id canary_otlog_v5_e18_remaining --z-indices 62 201
```

Do not start broad shards until all nine age-specific canary indices pass.
