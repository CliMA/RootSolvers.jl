
using RootSolvers
using Test

# Define test function
f(x) = x^2 - 4.0
f_df(x) = (x^2 - 4.0, 2x)

@testset "Method Type Defaults" begin
    # Test SecantMethod type with defaults
    sol = find_zero(f, SecantMethod, 1.0, 3.0)
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged

    # Test BisectionMethod type with defaults
    sol = find_zero(f, BisectionMethod, 1.0, 3.0)
    @test sol.root ≈ 2.0 atol = 1e-4

    # Test BrentsMethod type with defaults
    sol = find_zero(f, BrentsMethod, 1.0, 3.0)
    @test sol.root ≈ 2.0 atol = 1e-4

    # Test NewtonsMethod type with defaults
    sol = find_zero(f_df, NewtonsMethod, 1.0)
    @test sol.root ≈ 2.0 atol = 1e-4

    # Test NewtonsMethodAD type with defaults
    sol = find_zero(f, NewtonsMethodAD, 1.0)
    @test sol.root ≈ 2.0 atol = 1e-4

    # Test overrides work too
    tol = SolutionTolerance(1e-10)
    sol = find_zero(f, SecantMethod, 1.0, 3.0, CompactSolution(), tol)
    @test sol.root ≈ 2.0 atol = 1e-10
end

@testset "Direct Method Defaults" begin
    # Test SecantMethod with default tol
    # The default tol is SolutionTolerance(1e-4) for Float64
    sol = find_zero(f, SecantMethod(1.0, 3.0))
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged

    # Test BisectionMethod with default tol
    sol = find_zero(f, BisectionMethod(1.0, 3.0))
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged

    # Test RegulaFalsiMethod with default tol
    sol = find_zero(f, RegulaFalsiMethod(1.0, 3.0))
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged

    # Test BrentsMethod with default tol
    sol = find_zero(f, BrentsMethod(1.0, 3.0))
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged

    # Test NewtonsMethodAD with default tol
    sol = find_zero(f, NewtonsMethodAD(1.0))
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged

    # Test NewtonsMethod with default tol
    sol = find_zero(f_df, NewtonsMethod(1.0))
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged

    # Verify explicit nothing works same as default
    sol = find_zero(f, SecantMethod(1.0, 3.0), CompactSolution(), nothing)
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged
end
