# Tests for pre-evaluated endpoint residuals (`find_zero(f, M, x0, x1, y0, y1, ...)`),
# the `BracketedSolution` solution type, the `NoTolerance` criterion, and the
# `maxiters <= 0` endpoint behavior of the two-point methods.

const TWO_POINT_METHODS =
    (SecantMethod, BisectionMethod, RegulaFalsiMethod, BrentsMethod)
const BRACKETING_METHODS = (BisectionMethod, RegulaFalsiMethod, BrentsMethod)

@testset "Pre-evaluated endpoint residuals" begin
    for FT in (Float32, Float64)
        f(x) = x^2 - FT(4)
        x0, x1 = FT(0), FT(3)
        y0, y1 = f(x0), f(x1)
        for M in TWO_POINT_METHODS
            for soltype in (CompactSolution(), VerboseSolution())
                sol_std = find_zero(f, M, x0, x1, soltype)
                sol_pre = find_zero(f, M, x0, x1, y0, y1, soltype)
                # Identical trajectories: pre-evaluating endpoints must not change
                # the returned root or convergence status.
                @test sol_pre.root === sol_std.root
                @test sol_pre.converged === sol_std.converged
            end

            # The pre-evaluated form must save exactly the two endpoint evaluations.
            cnt_std = Ref(0)
            g_std(x) = (cnt_std[] += 1; x^2 - FT(4))
            find_zero(g_std, M, x0, x1)
            cnt_pre = Ref(0)
            g_pre(x) = (cnt_pre[] += 1; x^2 - FT(4))
            find_zero(g_pre, M, x0, x1, y0, y1)
            @test cnt_std[] == cnt_pre[] + 2
        end

        # Wrong pre-evaluated values are the caller's responsibility, but non-finite
        # ones must fail gracefully rather than iterate on garbage.
        for M in TWO_POINT_METHODS
            sol = find_zero(f, M, x0, x1, FT(NaN), y1)
            @test sol.converged === false
        end
    end
end

@testset "NoTolerance runs exactly maxiters iterations" begin
    for FT in (Float32, Float64)
        x0, x1 = FT(0), FT(3)
        for M in TWO_POINT_METHODS
            maxiters = 7
            cnt = Ref(0)
            g(x) = (cnt[] += 1; x^2 - FT(4))
            sol = find_zero(
                g, M, x0, x1, g(x0), g(x1),
                CompactSolution(), NoTolerance(), maxiters,
            )
            cnt[] -= 2 # remove the two explicit endpoint evaluations above
            # NoTolerance never signals convergence...
            @test sol.converged === false
            # ...and (except for Brent's independent exact-zero exit) the iteration
            # count is exactly maxiters.
            if M <: BrentsMethod
                @test cnt[] <= maxiters
            else
                @test cnt[] == maxiters
            end
            # The root estimate should nevertheless be accurate after 7 iterations
            # (bisection converges linearly: |error| <= (x1 - x0) / 2^maxiters).
            atol =
                M <: BisectionMethod ? (x1 - x0) / 2^maxiters :
                (FT == Float64 ? FT(1e-3) : FT(1e-2))
            @test abs(sol.root - 2) <= atol
        end
    end
end

@testset "BracketedSolution final state" begin
    for FT in (Float32, Float64)
        f(x) = x^2 - FT(4)
        x0, x1 = FT(0), FT(3)

        for M in BRACKETING_METHODS
            sol = find_zero(f, M, x0, x1, f(x0), f(x1), BracketedSolution())
            @test sol isa RootSolvers.BracketedSolutionResults
            @test isbits(sol)
            @test sol.converged === true
            # The final points must still bracket the root, contain it, and carry
            # consistent residuals.
            @test sol.y0 * sol.y1 <= 0
            @test min(sol.x0, sol.x1) <= sol.root <= max(sol.x0, sol.x1)
            @test sol.y0 === f(sol.x0)
            @test sol.y1 === f(sol.x1)
            @test sol.err === f(sol.root)

            # No sign change: inputs are returned unchanged with converged = false.
            sol_ns = find_zero(f, M, FT(3), FT(5), f(FT(3)), f(FT(5)), BracketedSolution())
            @test sol_ns.converged === false
            @test (sol_ns.x0, sol_ns.x1) === (FT(3), FT(5))
            @test sol_ns.root === FT(3) # smaller-residual endpoint
        end

        # Secant: the two points are the last iterates (no bracketing guarantee).
        sol_sec = find_zero(f, SecantMethod, x0, x1, f(x0), f(x1), BracketedSolution())
        @test sol_sec.converged === true
        @test sol_sec.root === sol_sec.x1

        # One-point methods do not carry a two-point state.
        @test_throws ArgumentError find_zero(
            f, NewtonsMethodAD{FT}(FT(1)), BracketedSolution(),
        )
    end

    # Pretty printing
    sol =
        find_zero(x -> x^2 - 4, RegulaFalsiMethod, 0.0, 3.0, -4.0, 5.0, BracketedSolution())
    str = sprint(show, sol)
    @test occursin("BracketedSolutionResults", str)
    @test occursin("Final points", str)
end

@testset "Two-point methods with maxiters <= 0" begin
    for FT in (Float32, Float64)
        f(x) = x^2 - FT(4)
        for M in TWO_POINT_METHODS
            # Previously an UndefVarError for the bracketing kernel; must return the
            # smaller-residual endpoint with converged = false.
            sol = find_zero(f, M, FT(0), FT(3), f(FT(0)), f(FT(3)),
                CompactSolution(), NoTolerance(), 0)
            @test sol.converged === false
            @test isfinite(sol.root)
            if M <: Union{BisectionMethod, RegulaFalsiMethod}
                @test sol.root === FT(0) # |f(0)| = 4 < |f(3)| = 5
            end
        end
    end
end

@testset "Pre-evaluated endpoints with dual numbers" begin
    # Differentiating through the solver must work when positions and residuals are
    # dual numbers supplied by the caller (as in solvers embedded in AD-transparent code).
    for M in BRACKETING_METHODS
        function h(θ)
            g(x) = x^2 - θ^2
            x0 = θ - 1
            x1 = θ + 1
            sol = find_zero(g, M, x0, x1, g(x0), g(x1))
            return sol.root^2
        end
        θ = 3.0
        deriv = ForwardDiff.derivative(h, θ)
        @test abs(deriv - 2 * θ) <= 2 * default_tol(Float64).tol
    end
end

# Function barrier with forced method-type specialization, mirroring the
# `find_zero_wrapper` pattern used by the allocation tests in `runtests.jl`.
function fval_alloc_measure(f::F, ::Type{M}, soltype::S) where {F, M, S}
    find_zero(f, M, 0.0, 3.0, -4.0, 5.0, soltype, NoTolerance(), 10) # compile
    return @allocated find_zero(f, M, 0.0, 3.0, -4.0, 5.0, soltype, NoTolerance(), 10)
end

@testset "Pre-evaluated endpoints do not allocate" begin
    f_alloc(x) = x^2 - 4.0
    for M in TWO_POINT_METHODS, soltype in (CompactSolution(), BracketedSolution())
        @test fval_alloc_measure(f_alloc, M, soltype) == 0
    end
end
