
using Test
using RootSolvers

@testset "Method Selector Tests" begin
    # Test function
    f(x) = x^2 - 2.0

    # Expected root
    root_val = sqrt(2.0)

    # Test each selector type
    @testset "Scalar Dispatch" begin
        # Secant
        sol = find_zero(
            f,
            SecantSelector(),
            1.0,
            2.0,
            CompactSolution(),
            SolutionTolerance(1e-4),
            100,
        )
        @test abs(sol.root - root_val) < 1e-4

        # Bisection
        sol = find_zero(
            f,
            BisectionSelector(),
            1.0,
            2.0,
            CompactSolution(),
            SolutionTolerance(1e-4),
            100,
        )
        @test abs(sol.root - root_val) < 1e-4

        # Regula Falsi
        sol = find_zero(
            f,
            RegulaFalsiSelector(),
            1.0,
            2.0,
            CompactSolution(),
            SolutionTolerance(1e-4),
            100,
        )
        @test abs(sol.root - root_val) < 1e-4

        # Brents
        sol = find_zero(
            f,
            BrentsSelector(),
            1.0,
            2.0,
            CompactSolution(),
            SolutionTolerance(1e-4),
            100,
        )
        @test abs(sol.root - root_val) < 1e-4

        # Newton AD (needs x0)
        sol = find_zero(
            f,
            NewtonsADSelector(),
            1.5,
            CompactSolution(),
            SolutionTolerance(1e-4),
            100,
        )
        @test abs(sol.root - root_val) < 1e-4

        # Newton (needs x0, f returns (val, deriv))
        f_deriv(x) = (x^2 - 2.0, 2x)
        sol = find_zero(
            f_deriv,
            NewtonsSelector(),
            1.5,
            CompactSolution(),
            SolutionTolerance(1e-4),
            100,
        )
        @test abs(sol.root - root_val) < 1e-4
    end

    @testset "Broadcasting" begin
        N = 10
        x0 = ones(N)
        x1 = ones(N) .* 2.0

        # Secant broadcasting
        # Note: We must wrap the selector in Ref() or treat it as a scalar to broadcast correctly 
        # against the arrays x0 and x1
        selector = SecantSelector()
        results = find_zero.(
            f,
            selector,
            x0,
            x1,
            CompactSolution(),
            SolutionTolerance(1e-4),
            100,
        )

        @test length(results) == N
        @test all(r -> abs(r.root - root_val) < 1e-4, results)

        # Mix of scalar and array args (Testing the selector passing specifically)
        results = find_zero.(
            f,
            SecantSelector(),
            x0,
            2.0,
            CompactSolution(),
            SolutionTolerance(1e-4),
            100,
        )
        @test length(results) == N
        @test all(r -> abs(r.root - root_val) < 1e-4, results)
    end

    @testset "Type Aliases" begin
        @test SecantSelector === MethodSelector{SecantMethod}
        @test NewtonsSelector === MethodSelector{NewtonsMethod}
        @test NewtonsADSelector === MethodSelector{NewtonsMethodAD}
        @test BrentsSelector === MethodSelector{BrentsMethod}
        @test BisectionSelector === MethodSelector{BisectionMethod}
        @test RegulaFalsiSelector === MethodSelector{RegulaFalsiMethod}
    end
end
