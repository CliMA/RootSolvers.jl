
using Test
using RootSolvers

@testset "Method Type Tests" begin
    # Test function
    f(x) = x^2 - 2.0

    # Expected root
    root_val = sqrt(2.0)

    # Test each method type
    @testset "Scalar Dispatch" begin
        # Secant
        sol = find_zero(
            f,
            SecantMethod,
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
            BisectionMethod,
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
            RegulaFalsiMethod,
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
            BrentsMethod,
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
            NewtonsMethodAD,
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
            NewtonsMethod,
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
        # Note: Method types broadcast as scalars (Ref(T)) effectively
        results =
            find_zero.(
                f,
                SecantMethod,
                x0,
                x1,
                CompactSolution(),
                SolutionTolerance(1e-4),
                100,
            )

        @test length(results) == N
        @test all(r -> abs(r.root - root_val) < 1e-4, results)

        # Mix of scalar and array args (Testing the type passing specifically)
        results =
            find_zero.(
                f,
                SecantMethod,
                x0,
                2.0,
                CompactSolution(),
                SolutionTolerance(1e-4),
                100,
            )
        @test length(results) == N
        @test all(r -> abs(r.root - root_val) < 1e-4, results)
    end
end
