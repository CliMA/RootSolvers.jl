
using RootSolvers
using Test

# Define test function
f(x) = x^2 - 4.0
f_df(x) = (x^2 - 4.0, 2x)

@testset "MethodSelector Defaults" begin
    # Test SecantSelector with defaults
    sol = find_zero(f, SecantSelector(), 1.0, 3.0)
    @test sol.root ≈ 2.0 atol = 1e-4
    @test sol.converged

    # Test BisectionSelector with defaults
    sol = find_zero(f, BisectionSelector(), 1.0, 3.0)
    @test sol.root ≈ 2.0 atol = 1e-4

    # Test BrentsSelector with defaults
    sol = find_zero(f, BrentsSelector(), 1.0, 3.0)
    @test sol.root ≈ 2.0 atol = 1e-4

    # Test NewtonsSelector with defaults
    sol = find_zero(f_df, NewtonsSelector(), 1.0)
    @test sol.root ≈ 2.0 atol = 1e-4

    # Test NewtonsADSelector with defaults
    sol = find_zero(f, NewtonsADSelector(), 1.0)
    @test sol.root ≈ 2.0 atol = 1e-4

    # Test overrides work too
    tol = SolutionTolerance(1e-10)
    sol = find_zero(f, SecantSelector(), 1.0, 3.0, CompactSolution(), tol)
    @test sol.root ≈ 2.0 atol = 1e-10
end
