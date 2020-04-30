using Test
using RootSolvers

@testset "RootSolvers - compact solution correctness" begin
    f(x) = x^2 - 100^2
    f′(x) = 2x
    for FT in [Float32, Float64]
        for tol in
            [ResidualTolerance(FT(1e-6)), SolutionTolerance(FT(1e-3)), nothing]
            sol = find_zero(
                f,
                FT(0.0),
                FT(1000.0),
                SecantMethod(),
                CompactSolution(),
                tol,
            )
            @test sol.converged
            @test sol.root isa FT
            @test sol.root ≈ 100

            sol = find_zero(
                f,
                FT(0.0),
                FT(1000.0),
                RegulaFalsiMethod(),
                CompactSolution(),
                tol,
            )
            @test sol.converged
            @test sol.root isa FT
            @test sol.root ≈ 100

            sol =
                find_zero(f, FT(1.0), NewtonsMethodAD(), CompactSolution(), tol)
            @test sol.converged
            @test sol.root isa FT
            @test sol.root ≈ 100

            sol = find_zero(
                f,
                f′,
                FT(1.0),
                NewtonsMethod(),
                CompactSolution(),
                tol,
            )
            @test sol.converged
            @test sol.root isa FT
            @test sol.root ≈ 100
        end
    end
end

@testset "RootSolvers - compact solution non-converged cases" begin
    f(x) = x^2 - 100^2
    f′(x) = 2x
    for FT in [Float32, Float64]
        sol = find_zero(
            f,
            FT(0.0),
            FT(1000.0),
            SecantMethod(),
            CompactSolution(),
            SolutionTolerance(FT(1e-1)),
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
            SolutionTolerance(FT(1e-1)),
            1,
        )
        @test !sol.converged
        @test sol.root isa FT

        sol = find_zero(
            f,
            FT(1.0),
            NewtonsMethodAD(),
            CompactSolution(),
            SolutionTolerance(FT(1e-1)),
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
            SolutionTolerance(FT(1e-1)),
            1,
        )
        @test !sol.converged
        @test sol.root isa FT
    end
end

@testset "RootSolvers - verbose solution correctness" begin
    f(x) = x^2 - 100^2
    f′(x) = 2x
    for FT in [Float32, Float64]
        sol =
            find_zero(f, FT(0.0), FT(1000.0), SecantMethod(), VerboseSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ 100
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
        @test sol.root ≈ 100
        @test sol.err < 1e-3
        @test sol.iter_performed < 20
        @test sol.iter_performed + 1 ==
              length(sol.root_history) ==
              length(sol.err_history)

        sol = find_zero(f, FT(1.0), NewtonsMethodAD(), VerboseSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ 100
        @test sol.err < 1e-3
        @test sol.iter_performed < 20
        @test sol.iter_performed + 1 ==
              length(sol.root_history) ==
              length(sol.err_history)

        sol = find_zero(f, f′, FT(1.0), NewtonsMethod(), VerboseSolution())
        @test sol.converged
        @test sol.root isa FT
        @test sol.root ≈ 100
        @test sol.err < 1e-3
        @test sol.iter_performed < 20
        @test sol.iter_performed + 1 ==
              length(sol.root_history) ==
              length(sol.err_history)
    end
end
