using Test
using RootSolvers

f(x) = x^2 - 100^2
f′(x) = 2x

# True root
x̂ = 100
@assert f(x̂) ≈ 0

@testset "RootSolvers - compact solution correctness" begin
    for FT in [Float32, Float64]
        sol =
            find_zero(f, FT(0.0), FT(1000.0), SecantMethod(), CompactSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ x̂

        sol = find_zero(
            f,
            FT(0.0),
            FT(1000.0),
            RegulaFalsiMethod(),
            CompactSolution(),
        )
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ x̂

        sol = find_zero(f, FT(1.0), NewtonsMethodAD(), CompactSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ x̂

        sol = find_zero(f, f′, FT(1.0), NewtonsMethod(), CompactSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ x̂
    end
end

@testset "RootSolvers - compact solution non-converged cases" begin
    for FT in [Float32, Float64]
        sol = find_zero(
            f,
            FT(0.0),
            FT(1000.0),
            SecantMethod(),
            CompactSolution(),
            FT(1e-1),
            1,
        )
        @test !sol.converged
        @test sol.root isa FT

        sol = find_zero(
            f,
            FT(0.0),
            FT(1000.0),
            RegulaFalsiMethod(),
            CompactSolution(),
            FT(1e-1),
            1,
        )
        @test !sol.converged
        @test sol.root isa FT

        sol = find_zero(
            f,
            FT(1.0),
            NewtonsMethodAD(),
            CompactSolution(),
            FT(1e-1),
            1,
        )
        @test !sol.converged
        @test sol.root isa FT

        sol = find_zero(
            f,
            f′,
            FT(1.0),
            NewtonsMethod(),
            CompactSolution(),
            FT(1e-1),
            1,
        )
        @test !sol.converged
        @test sol.root isa FT
    end
end

@testset "RootSolvers - verbose solution correctness" begin
    for FT in [Float32, Float64]
        sol =
            find_zero(f, FT(0.0), FT(1000.0), SecantMethod(), VerboseSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ x̂
        @test sol.err < 1e-3
        @test sol.iter_performed < 20
        @test sol.iter_performed + 1 ==
              length(sol.root_history) ==
              length(sol.err_history)

        sol = find_zero(
            f,
            FT(0.0),
            FT(1000.0),
            RegulaFalsiMethod(),
            VerboseSolution(),
        )
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ x̂
        @test sol.err < 1e-3
        @test sol.iter_performed < 20
        @test sol.iter_performed + 1 ==
              length(sol.root_history) ==
              length(sol.err_history)

        sol = find_zero(f, FT(1.0), NewtonsMethodAD(), VerboseSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ x̂
        @test sol.err < 1e-3
        @test sol.iter_performed < 20
        @test sol.iter_performed + 1 ==
              length(sol.root_history) ==
              length(sol.err_history)

        sol = find_zero(f, f′, FT(1.0), NewtonsMethod(), VerboseSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ x̂
        @test sol.err < 1e-3
        @test sol.iter_performed < 20
        @test sol.iter_performed + 1 ==
              length(sol.root_history) ==
              length(sol.err_history)
    end
end
