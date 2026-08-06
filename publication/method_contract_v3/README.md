# FEAST publication method-contract v3 addendum

This additive contract layer preserves `../method_contract_v2/` byte-for-byte
and binds the current clean rerun evidence. It exists because the v2 layer is a
frozen 2026-07-19 record: its three conditional-OT entries describe the legacy
noncanonical outputs, and its Study 00 binding predates the later pair-support
and scCube Slide-seq repairs.

The v3 addendum contains exactly eight article workflows. Each row pins its v2
predecessor by SHA-256, binds current evidence directly, states the remaining
claim limits, and keeps release/figure/claim authorization false. It does not
rewrite history or make the repaired FEAST 1.1.0 candidate a release.

Validate from the repository root:

```bash
python publication/method_contract_v3/validate_contract.py
```

Study 00 remains scientifically blocked because its ignored `metrics_v2`
table was modified after being frozen and the original bytes are unavailable;
the new final table is directly hash-bound, but its full transitive source
lineage cannot be restored. Studies 05–07 refer to clean-study numbering while
retaining their legacy article workflow IDs.
