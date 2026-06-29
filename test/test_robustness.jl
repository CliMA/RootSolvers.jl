# Regression tests for the Newton line-search, bracketing convergence, and default
# tolerance fixes (branch ts/newton-fixes). These run on the CPU `Array` path only.

using Test
using RootSolvers
using StaticArrays
using ForwardDiff

@testset "Newton recovers from non-finite overshoot (#1)" begin
    # Barrier function: f(x) = 1/x - 1 for x > 0, and a non-finite value otherwise;
    # the root is at x = 1. The full Newton step from x0 = 2 lands exactly on x = 0,
    # where f is non-finite. The backtracking line search must shrink the step to
    # recover, rather than aborting on the first non-finite trial point.
    for bad in (Inf, NaN)
        f = x -> x > 0 ? 1 / x - 1 : oftype(x, bad)
        ff = x -> x > 0 ? (1 / x - 1, -1 / x^2) : (oftype(x, bad), oftype(x, bad))
        for (name, method, fn) in (
            ("NewtonsMethodAD", NewtonsMethodAD{Float64}(2.0), f),
            ("NewtonsMethod", NewtonsMethod{Float64}(2.0), ff),
        )
            sol = find_zero(fn, method, CompactSolution())
            @test sol.converged
            @test isapprox(sol.root, 1.0; atol = 1e-3)
        end
    end

    # When the step direction is itself non-finite and no finite trial can be found,
    # the solver must still terminate gracefully (no hang, no throw) as not-converged.
    f_bad = x -> oftype(x, NaN)
    sol = find_zero(f_bad, NewtonsMethodAD{Float64}(1.0), CompactSolution())
    @test sol.converged === false
end

@testset "Newton's singular-derivative fallback is inline (#1b)" begin
    # f(x) = (x-2)^2 - 1 has roots at x = 1 and x = 3 and a critical point at x = 2,
    # where f′ = 0 exactly. Starting Newton there makes Δx = f/f′ non-finite, so the
    # singular-step fallback fires on the very first iteration. The fallback runs
    # inline (a finite-difference slope, same loop).
    for (method, fn) in (
        (NewtonsMethodAD{Float64}(2.0), x -> (x - 2)^2 - 1),
        (NewtonsMethod{Float64}(2.0), x -> ((x - 2)^2 - 1, 2 * (x - 2))),
    )
        sol = find_zero(fn, method, CompactSolution())
        @test sol.converged
        @test abs((sol.root - 2)^2 - 1) <= 1e-3   # converged to an actual root
    end

    # Because the fallback is inline, the VerboseSolution iteration history is preserved 
    sol_v =
        find_zero(x -> (x - 2)^2 - 1, NewtonsMethodAD{Float64}(2.0), VerboseSolution())
    @test sol_v.converged
    @test length(sol_v.root_history) > 1          # fallback step(s) recorded, not dropped
    @test isfinite(sol_v.err)
end

@testset "Newton is scale-invariant near small derivatives (#2)" begin
    # Scaling f by a constant scales the derivative but not the Newton step Δx = f/f′,
    # so the iteration must behave identically across scales.
    for k in (1.0, 1e-8, 1e8)
        f = x -> k * (x^2 - 4)
        ff = x -> (k * (x^2 - 4), k * 2x)
        for (method, fn) in (
            (NewtonsMethodAD{Float64}(1.0), f),
            (NewtonsMethod{Float64}(1.0), ff),
        )
            sol = find_zero(fn, method, CompactSolution())
            @test sol.converged
            @test isapprox(sol.root, 2.0; atol = 1e-3)
        end
    end

    # A high-multiplicity root has a vanishing derivative near the root, yet Newton
    # must still converge: Δx = (x-3)^5 / (5 (x-3)^4) = (x-3)/5 stays finite, so the
    # singular-step fallback does not trigger.
    sol = find_zero(
        x -> (x - 3)^5,
        NewtonsMethodAD{Float64}(1.0),
        CompactSolution(),
        SolutionTolerance{Float64}(1e-6),
        10_000,
    )
    @test sol.converged
    @test isapprox(sol.root, 3.0; atol = 1e-2)
end

@testset "Regula Falsi converges on the step, not bracket width (#3)" begin
    # f(x) = x^3 - 1, root x = 1. On [0, 5] Regula Falsi keeps the upper endpoint (5)
    # fixed and creeps the lower endpoint up, so the bracket width stays ~4 and never
    # meets a width-based tolerance. With step-based convergence it stops promptly at
    # the requested SolutionTolerance instead of grinding down to machine precision.
    f = x -> x^3 - 1
    tol = SolutionTolerance{Float64}(1e-3)
    sol = find_zero(
        f,
        RegulaFalsiMethod{Float64}(0.0, 5.0),
        VerboseSolution(),
        tol,
        10_000,
    )
    @test sol.converged
    @test isapprox(sol.root, 1.0; atol = 1e-2)              # within a few × tol
    @test sol.iter_performed < 50                           # not grinding to machine eps
    # Convergence fired on the inter-iterate step being below tol:
    @test abs(sol.root_history[end] - sol.root_history[end - 1]) < tol.tol

    # The fix must not break the well-behaved cases: Bisection still converges
    # accurately, and an exact midpoint hit is detected.
    sol_b = find_zero(x -> x^2 - 4, BisectionMethod{Float64}(0.0, 5.0))
    @test sol_b.converged
    @test isapprox(sol_b.root, 2.0; atol = 1e-3)
end

@testset "default_tol is consistent across scalar / array / dual (#4)" begin
    for (FT, val) in ((Float64, 1e-4), (Float32, 1.0f-3))
        sc = default_tol(FT)
        @test sc isa SolutionTolerance{FT}
        @test sc.tol == val
        # Arrays and dual numbers over the same element type share the default
        @test default_tol(Vector{FT}) isa SolutionTolerance{FT}
        @test default_tol(Vector{FT}).tol == val
        @test default_tol(SVector{3, FT}).tol == val
        @test default_tol(ForwardDiff.Dual{Nothing, FT, 2}).tol == val
    end
end
