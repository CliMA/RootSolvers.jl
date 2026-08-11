# Tests for tracing the solver kernels under Reactant: scalar `@jit` solves and
# broadcasting over `to_rarray` arrays, compared against the plain-Julia (untraced) result.
# `NewtonsMethodAD` is excluded — `ForwardDiff.Dual` cannot wrap a `TracedRNumber`, an
# upstream limitation orthogonal to this package's `@trace` support.
#
# Each test constructs a *fresh* `to_rarray` input rather than reusing one across multiple
# `@jit` calls: a compiled call may donate/alias a scalar input's buffer as internal loop
# storage, so reusing the same traced array across calls with different arguments can
# observe a stale/corrupted value (a Reactant calling-convention footgun, not a RootSolvers
# bug — see the `copy(x0)` guards at the top of each `_find_zero_*` kernel).

using Reactant

const REACTANT_TWO_POINT_METHODS =
    (SecantMethod, BisectionMethod, RegulaFalsiMethod, BrentsMethod)

f_quad(x) = x^2 - 4
fdf_quad(x) = (x^2 - 4, 2x)

rscalar(x) = Reactant.to_rarray(x; track_numbers = Number)

@testset "Reactant: scalar solve matches plain Julia" begin
    for M in REACTANT_TWO_POINT_METHODS
        cpu = find_zero(f_quad, M, 0.0, 3.0, CompactSolution())
        traced = @jit find_zero(f_quad, M, rscalar(0.0), rscalar(3.0), CompactSolution())
        @test Float64(traced.root) ≈ cpu.root
        @test Bool(traced.converged) == cpu.converged
    end

    cpu = find_zero(fdf_quad, NewtonsMethod, 2.0, CompactSolution())
    traced = @jit find_zero(fdf_quad, NewtonsMethod, rscalar(2.0), CompactSolution())
    @test Float64(traced.root) ≈ cpu.root
    @test Bool(traced.converged) == cpu.converged
end

@testset "Reactant: TwoPointSolution matches plain Julia" begin
    for M in REACTANT_TWO_POINT_METHODS
        cpu = find_zero(f_quad, M, 0.0, 3.0, TwoPointSolution())
        traced = @jit find_zero(f_quad, M, rscalar(0.0), rscalar(3.0), TwoPointSolution())
        @test Float64(traced.root) ≈ cpu.root
        @test Bool(traced.converged) == cpu.converged
        @test Float64(traced.x0) ≈ cpu.x0
        @test Float64(traced.x1) ≈ cpu.x1
    end
end

@testset "Reactant: broadcast matches the CPU loop" begin
    # Broadcasting `find_zero` itself doesn't work under Reactant (no `similar` for an
    # array of results structs), so broadcast a scalar-returning wrapper instead. On the
    # CPU, ordinary `find_zero.(...)` broadcasting (array of structs) works directly.
    root_only(x0, x1) = find_zero(f_quad, SecantMethod, x0, x1, CompactSolution()).root
    xs = collect(0.5:0.3:3.0)
    cpu_sols = find_zero.(f_quad, SecantMethod, xs, xs .+ 1, CompactSolution())
    cpu_roots = getfield.(cpu_sols, :root)
    traced_roots =
        Array(@jit root_only.(Reactant.to_rarray(xs), Reactant.to_rarray(xs .+ 1)))
    @test traced_roots ≈ cpu_roots
end

@testset "Reactant: non-bracketing interval and non-finite guess" begin
    # `SecantMethod` doesn't require a bracket, so it may legitimately converge on an
    # interval that doesn't bracket a root — compare against the CPU result rather than
    # assuming `converged == false` for every method.
    for M in REACTANT_TWO_POINT_METHODS
        cpu = find_zero(f_quad, M, 1.0, 1.5, CompactSolution())
        traced = @jit find_zero(f_quad, M, rscalar(1.0), rscalar(1.5), CompactSolution())
        @test Bool(traced.converged) == cpu.converged
        @test Float64(traced.root) ≈ cpu.root

        cpu_nan = find_zero(f_quad, M, NaN, 3.0, CompactSolution())
        traced_nan =
            @jit find_zero(f_quad, M, rscalar(NaN), rscalar(3.0), CompactSolution())
        @test Bool(traced_nan.converged) == cpu_nan.converged
        @test isnan(Float64(traced_nan.root)) == isnan(cpu_nan.root)
    end

    cpu_inf = find_zero(fdf_quad, NewtonsMethod, Inf, CompactSolution())
    traced_inf = @jit find_zero(fdf_quad, NewtonsMethod, rscalar(Inf), CompactSolution())
    @test Bool(traced_inf.converged) == cpu_inf.converged
    @test Float64(traced_inf.root) ≈ cpu_inf.root
end

@testset "Reactant: NoTolerance runs exactly maxiters iterations" begin
    for M in REACTANT_TWO_POINT_METHODS
        for m in 1:6
            cpu = find_zero(f_quad, M, 0.0, 3.0, CompactSolution(), NoTolerance(), m)
            traced = @jit find_zero(
                f_quad, M, rscalar(0.0), rscalar(3.0), CompactSolution(), NoTolerance(),
                m,
            )
            @test Float64(traced.root) ≈ cpu.root
            @test Bool(traced.converged) == false
        end
    end
end
