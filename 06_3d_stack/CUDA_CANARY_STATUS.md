# Study 06 CUDA canary status

Updated: 2026-07-20
Status: authorized by passed v5 preflight; not yet executed

No Study 06 CUDA generation was launched. The package-level CUDA log-domain
gate passed. Configuration v5 declares the independently reproduced
medium-density assignment randomness `0.30`, and its fresh full preflight
passed with 147 inputs and 93 target assignments frozen under
`outputs/preflight_otlog110_v5_20260720/`.

The first bounded generation checks are one original target from each density:

```bash
python run.py --output-dir outputs/preflight_otlog110_v5_20260720 --gap 3 --target-id 5
python run.py --output-dir outputs/preflight_otlog110_v5_20260720 --gap 5 --target-id 6
python run.py --output-dir outputs/preflight_otlog110_v5_20260720 --gap 10 --target-id 6
```

They must use the pinned CUDA environment and retain exact solver/device/dtype
diagnostics. Do not substitute CPU generation or relax the preflight gate.
The canaries remain pending because no further workflow-level CUDA launch was
available during this validation session; this is not a FEAST numerical
failure or an output failure.
