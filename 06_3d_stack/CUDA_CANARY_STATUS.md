# Study 06 CUDA canary status

Updated: 2026-07-31
Status: passed; production, validation, and evaluation complete

The package-level CUDA log-domain gate and all three Study 06 canaries passed
with the pinned FEAST 1.1.0 candidate on the NVIDIA RTX A6000. Configuration
v5 declares the independently reproduced medium-density assignment randomness
`0.30`. Its fresh full preflight froze 147 inputs and 93 target assignments.

The first bounded generation checks are one original target from each density:

```bash
python run.py --output-dir outputs/preflight_otlog110_v5_20260720 --gap 3 --target-id 5
python run.py --output-dir outputs/preflight_otlog110_v5_20260720 --gap 5 --target-id 6
python run.py --output-dir outputs/preflight_otlog110_v5_20260720 --gap 10 --target-id 6
```

They used the pinned CUDA environment and retained exact
solver/device/dtype diagnostics. Production then completed 93/93 outputs. The
independent validation summary reports `status=passed` and
`all_positive_convergence=true`; its SHA-256 is
`c41bb104a06a6711070f1888533422ba11596d7e7bebf4fa044bc1aaeabf4a37`.
The evaluation provenance SHA-256 is
`2bb9d66f4fc3ad20c4e9683ba7fc5db4dedbb52c86b06ed515e1e9d5cea6c85b`.
This closes the execution gate but does not promote an article claim.
