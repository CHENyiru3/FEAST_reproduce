# FEAST publication rerun status

Checkpoint: 2026-07-19 01:57 CDT / 2026-07-19 06:57 UTC.

This is an execution checkpoint, not a final publication disposition. FEAST
remains provisional version 1.0.2 at commit
`68816e5c1862a6fa2a49bc30609d617c7fa4b449`. No release commit, tag, or PyPI
publication has been created.

## Cross-study gates

- Supported wheel environment: Python 3.11.15, NumPy 1.26.4.
- Installed-wheel tests: 92 passed; `pip check` passed.
- RNG article gate: passed, with three identical output matrix hashes.
- Repository integrity checker: passed after the current Study 01 work.
- External method environments remain isolated and do not expand FEAST's
  supported NumPy range.

## Study status

| Study | Completed fresh evidence | Remaining work |
|---|---|---|
| 00 | 12/12 FEAST simulations; 60/60 atomic metric rows; historical comparison and validation | Final cross-study integration only |
| 01 | 81/81 FEAST simulations; 84/84 fixed panels; 15/15 combined-output direction audits; 84/84 GraphST, STAGATE+mclust, and publication-valid Leiden outputs; 252/252 metrics; full validation and historical comparisons | Final cross-study integration only |
| 02 | 20/20 centered rotation inputs; 20/20 PASTE outputs with positive convergence evidence | 20 Spateo CUDA jobs; combined alignment scoring and decision |
| 03 | 6/6 FEAST simulations; 1/6 RCTD outputs accepted | 5 RCTD and 6 Cell2location jobs; common-support scoring and decision |
| 04 | Fresh inputs prepared; 12/12 GraphST outputs and 1/12 STAMP outputs accepted | 11 STAMP and 12 scVI CUDA jobs; 60-row atomic report and validation |

## Study 01 checkpoint

The simulation manifest contains 81 successful rows, 27 per slice, and has
SHA-256
`2fa032367383f7369c33f14e9adeed315ceadb02a1443a4349a1864c8807f254`.
The resume operation hash-verified and skipped the original 69 candidates;
their manifest rows remained byte-identical.

Heavy-tailed fitted-distribution moment diagnostics printed unstable values
for some combined alterations. A separate matrix-level audit verified all 15
saved combined-condition matrices as finite, nonnegative integer counts with
exact spot/gene order and the expected directional effects. That audit is
bound to the simulation manifest and is not a replacement for it.

The fixed-panel manifest contains 84 rows and has SHA-256
`0316c33b91de5fde8d18b7223631176294e3cae97d32400540a5348f6d2b6dcd`.
Exactly one panel, `151670/sparsity_pos_1`, contains one zero-count spot after
restriction to the raw-derived 3,000-gene panel. The spot is retained to
preserve exact support; publication-valid Leiden completed successfully for
that panel.

Fresh GraphST, STAGATE+mclust, and label-free Leiden each completed 84/84
jobs. GraphST and STAGATE have positive per-job CUDA allocation evidence.
Their run-manifest SHA-256 values are, respectively,
`758652a1f07a414653aac7490a3acd644ce8f8e3d39ae3c42f1d7b0283c73141`
and `8b21309cf956d0d75adb6689026c63687d0470446af32b3fddd807cc0a297ae6`.
The label-free Leiden run-manifest SHA-256 is
`34db79052c3f04f52236df11919d4b6319977c044f3a2ef59491ab66d82432c2`.

Across all 81 FEAST simulations, mean ARI/NMI/AMI are:

- GraphST: 0.42969/0.55977/0.55885;
- STAGATE+mclust: 0.37401/0.50515/0.50400;
- label-free Leiden: 0.21273/0.30421/0.30240.

FEAST-baseline minus real-baseline ARI/NMI/AMI differences are
-0.00770/+0.00786/+0.00786 for GraphST,
+0.00524/+0.00634/+0.00637 for STAGATE+mclust, and
-0.00023/-0.00157/-0.00141 for Leiden. The maximum absolute gap is 0.00786
after rounding. The complete 252-row metric table has SHA-256
`10abbadbcf8c26d2b85e92a3357a8380fe4f4bc46882e6d42d2388a3d1882fab`.

The end-to-end validator passed all 81 simulations, 84 panels, and 252 method
outputs. Exact-key comparisons cover all 243 rows in the historical
independent-HVG table and all 252 rows in the prior repaired fixed-panel
table. Numerical deltas are classified as mixed workflow changes because
fixed panels, fresh simulations, and stochastic method reruns changed
together; Leiden additionally changed to label-free selection. The
baseline-matching headline direction is unchanged, and Study 01 is now a
validated publication candidate. Overall release authorization remains
blocked on the other studies and final gates.

The first Leiden scoring root is preserved as noncanonical because a guard
mistook the substring `ari` inside `modularity_gamma1` for an ARI field. It
produced no metric rows. The corrected report uses a new `_v2` root.

## CUDA execution

The managed execution service now permits approved CUDA Python launches.
Study 01 GraphST and STAGATE+mclust ran on the NVIDIA RTX A6000 and recorded
positive allocation evidence for every production job. Their external method
environments use NumPy 1.23.x, which is outside FEAST's supported range;
FEAST was not imported or executed in those workers.

## Next actions

1. Resume Study 02 Spateo on CUDA.
2. Resume Study 03 so the remaining RCTD and Cell2location jobs complete.
3. Resume Study 04 STAMP, then scVI, on CUDA.
4. Score and validate each complete study, generate old-versus-new tables, and
   update canonical decisions and the figure-source map.
5. Run final clean-environment tests/builds, core hashes, artifact freeze, and
   all eight method contracts. Keep release disposition blocked until these
   checks pass.
