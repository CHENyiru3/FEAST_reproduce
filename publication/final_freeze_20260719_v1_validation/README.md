# Final freeze v1 independent validation

The independent validator reread and SHA-256-checked all 3,222 manifested
files, totaling 24,988,647,813 bytes. It rebuilt the declared inventories from
the source roots, compared every path, classification, size, and digest,
rechecked 3,132 embedded references, and reproduced the decision, contract,
configuration-ID, seed, and no-release gates.

Validation passed with exit code 0. This accepts the evidence snapshot only;
the scientific disposition remains blocked, and no release, tag, claim, or
publication is authorized.

Reproduce from the repository root with the supported Python environment:

```bash
python publication/validate_final_freeze.py \
  --repo REPRO=. \
  --repo PKG=../FEAST \
  --repo EXP=../FEAST_experiments \
  --freeze-root publication/final_freeze_20260719_v1
```
