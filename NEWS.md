# RootSolvers.jl Release Notes

main
-------

v1.1.0
-------

- Added pre-evaluated endpoint support for the two-point methods: `find_zero(f, MethodType,
  x0, x1, y0, y1, ...)` accepts the endpoint residuals `y0 = f(x0)`, `y1 = f(x1)` so
  if they were already evaluated by the caller (e.g. while scanning for a bracket), they
  can be reused, avoiding redundant function evaluations. Supported for `SecantMethod`,
  `BisectionMethod`, `RegulaFalsiMethod`, and `BrentsMethod`.
- Added `TwoPointSolution`, a GPU-compatible solution type that additionally returns the
  final two-point state `(x0, x1, y0, y1)` of the solver, enabling caller-side convergence
  checks and post-processing.
- Added `NoTolerance`, a convergence criterion to force the solver to run for exactly
  `maxiters` iterations with no data-dependent early exit. This gives uniform per-point
  work in fixed-iteration GPU workloads.
- In the shared bracketing kernel (bisection/regula falsi), a converged solution now
  reports the bracket state *after* incorporating the accepted iterate (relevant only for
  `TwoPointSolution`); returned roots and convergence flags are unchanged.
- Fixed an `UndefVarError` in the bracketing methods when called with `maxiters = 0`
  (or negative, which shouldn't be used); they now return the smaller-residual endpoint with `converged = false` (or as assessed by the tolerance).

v1.0.3
-------

- Improved robustness of Newton's method by implementing an inline finite-difference fallback for singular derivatives. This avoids the loss of iteration history and improves performance by keeping the kernel a single iteration loop.
- Replaced short-circuiting branches (`||`) with bitwise operators (`|`) in tolerance checks and solver loops to improve GPU performance and reduce warp divergence.
