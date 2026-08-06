# Study 07 CUDA canary status

Updated: 2026-07-31
Status: passed; complete production scope validated

The installed FEAST candidate passed focused CUDA log-domain parity tests, and
exact audits completed both the former E18.5 failure block and an E15.5
24,994,980-pair block on the RTX A6000. All nine declared canary indices then
passed with positive convergence evidence; the workflow proceeded without a
CPU or method fallback.

The bounded gates were run from this directory with the pinned CUDA interpreter
declared in `../environments/README.md`:

```bash
"$FEAST_PY" run.py --age E15.5 --shard-id canary_otlog_v5_e15_z12 --z-indices 12
"$FEAST_PY" run.py --age E18.5 --shard-id canary_otlog_v5_e18_edge_middle --z-indices 0 100
```

Validate their per-slice records, hashes, exact identity, and positive solver
evidence before expanding to the complete declared canary sets:

```bash
"$FEAST_PY" run.py --age E15.5 --shard-id canary_otlog_v5_e15_remaining --z-indices 0 78 47 157
"$FEAST_PY" run.py --age E18.5 --shard-id canary_otlog_v5_e18_remaining --z-indices 62 201
```

All canaries passed before broad shards were started. Production,
consolidation, and independent validation are complete: 158/158 E15.5 and
202/202 E18.5 levels, exact blueprint identity, and 12,290/12,290 converged
transport records. The final validation SHA-256 is
`2e9a1607b8e21e1e035c2b8353a0be78bcf96782a58ba7266d0138183b17aa1f`.
This closes the CUDA execution gate; article promotion remains a separate
author decision.
