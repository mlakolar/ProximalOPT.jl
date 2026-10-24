# Sparse and sparse-plus-low-rank IHT comparisons

This bounded benchmark addresses the remaining comparisons of
[issue #24](https://github.com/mlakolar/ProximalSolvers.jl/issues/24).
It does not rerun the already measured ProximalAlgorithms comparison:
[PR #53](https://github.com/mlakolar/ProximalSolvers.jl/pull/53) and
[PR #50](https://github.com/mlakolar/ProximalSolvers.jl/pull/50) record those
Discovery measurements and their adverse findings, including LiLin's failed
fallback and different nonconvex local minima.

`legacy.jl` calls the original CovarianceSelection routines at the issue's
frozen revision `edba5de`. It measures all three routines: plain D-trace IHT,
latent D-trace IHT and latent log-det IHT. A recorded fixed-step grid
`0.05, 0.2, 0.5`, scaled by the relevant curvature, provides a bounded tuning
comparison. Every grid result remains visible, including exceptions and
nonfinite objectives. The legacy routines are not rewritten in this harness.

The legacy preparation process saves its exact initializers as plain arrays.
The native process receives those same points and minimizes the same smooth
objectives, using a symmetric sparse block and, for a latent model, a second
factor block `U`. It compares plain proximal gradient, normalized proximal
gradient and monotone APG. A separate centered sparse-regression case also
compares normalized steps with support debiasing. The native coupled losses
are self-contained benchmark fixtures, with a Cholesky check before every
log-det inverse. They are not new estimator APIs.

Both runners compile each method/problem shape before timing. Native method
order reverses between repetitions. Legacy and native runners are separate
processes; their timing blocks are not interleaved across packages. These
small timings alone must not be presented as an exhaustive speed comparison.
The default dimensions are `10,20`, rank `2`, two repetitions and at most
1,000 iterations. Native solves additionally have a 30-second soft limit.

`native_vs_legacy.tsv` compares every returned native solution with the best
finite, feasible legacy objective over the recorded fixed-step grid. Reference
selection checks symmetry, the sparse support budget, factor dimensions and
positive definiteness where the log-det objective requires it. Ineligible
legacy solutions remain in `legacy_reference_rejections.tsv`; a missing valid
reference produces an explicit unavailable row. It records both objective and
support agreement; failures and support differences are not repaired or
discarded. Raw solver and warmup failures remain in the runners' TSV files.
The new exact symmetric projection can select a different support
from the legacy magnitude threshold when diagonal entries are constrained.
These are nonconvex local solutions, so neither objective ordering nor matching
supports is a universal guarantee. `legacy.tsv` and `native.tsv` also retain
the maximum objective increase, finite-trace status and final positive
definiteness. Inspect these fields before making a monotonicity claim.

## Discovery recipe

All setup, data generation, execution and summarization run under Slurm.
Bind the legacy environment to CovarianceSelection `edba5de`, the native
environment to the new ProximalSolvers checkout, and both to the recorded
ProximalBase snapshot. The benchmark environment requires ProximalBase 0.5.3
or later. Separate processes exchange only plain arrays serialized by the same
Julia version.

```sh
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
JULIA=/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia
SOURCE=/absolute/path/to/new/source
OUT=/absolute/path/to/run/results
# Each command below runs in the Slurm allocation.
srun --exclusive --exact -N1 -n1 -c1 "$JULIA" --startup-file=no \
  --project=/absolute/path/to/legacy-env \
  "$SOURCE/benchmark/nonconvex/legacy.jl" --root "$OUT" --prepare --sizes 10,20
srun --exclusive --exact -N1 -n1 -c1 "$JULIA" --startup-file=no \
  --project=/absolute/path/to/legacy-env \
  "$SOURCE/benchmark/nonconvex/legacy.jl" --root "$OUT" --rounds 2
srun --exclusive --exact -N1 -n1 -c1 "$JULIA" --startup-file=no \
  --project=/absolute/path/to/native-env \
  "$SOURCE/benchmark/nonconvex/native.jl" --root "$OUT" --rounds 2
srun --exclusive --exact -N1 -n1 -c1 "$JULIA" --startup-file=no \
  "$SOURCE/benchmark/nonconvex/compare.jl" --root "$OUT"
```

`mendel.jl` is an optional reference runner in the separate `mendel/`
environment. Its API follows the
[official MendelIHT source](https://github.com/OpenMendel/MendelIHT.jl/blob/master/src/fit.jl).
It uses the same centered regression arrays, sparsity and initial-zero option,
with and without MendelIHT debiasing. Both branches warm before timing, and a
failed warmup is recorded explicitly. MendelIHT 1.4.11 requires
`memory_efficient=false` for debiasing; the adapter uses this setting for the
debiased branch and retains the default `true` for the other branch. Earlier
Discovery runs with both branches using the default recorded the debiasing
error, "Currently debiasing only works with memory_efficient=false". Those
failed rows remain part of the run record. The implicit intercept has its optimum
at zero on these centered inputs. Returned coefficients are checked through
their residual objective and support, rather than comparing likelihood
normalizations. A missing package or adapter failure remains an explicit
unavailable/error row; it is not a successful comparison. Dependency setup
belongs inside Slurm and may be omitted when it exceeds the run's budget.

```sh
srun --exclusive --exact -N1 -n1 -c1 "$JULIA" --startup-file=no \
  --project="$SOURCE/benchmark/nonconvex/mendel" \
  "$SOURCE/benchmark/nonconvex/mendel.jl" --root "$OUT" --rounds 2
```

Retain the raw TSVs, support/solution records, source revisions and Slurm
output. Benchmark outcomes are pending until the Discovery run is inspected.
