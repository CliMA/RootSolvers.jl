push!(LOAD_PATH, joinpath(@__DIR__, "..", "env", "test"))
using Test
using RootSolvers
using StaticArrays

f(x) = x^2 - 100^2
f′(x) = 2x
x̃ = 100

@testset "Scalar correctness test with compact solution" begin
    F′ = typeof(f′)
    for FT in [Float32, Float64]
        for tol in
            [ResidualTolerance{FT}(1e-6), SolutionTolerance{FT}(1e-3), nothing]
            for method in [
                SecantMethod{FT}(0.0, 1000.0),
                RegulaFalsiMethod{FT}(0.0, 1000.0),
                NewtonsMethodAD{FT}(1.0),
                NewtonsMethod{FT, F′}(1.0, f′),
            ]
                sol = find_zero(f, method, CompactSolution(), tol)
                @test isbits(method)
                @test sol.converged
                @test sol.root isa FT
                @test sol.root ≈ x̃
            end
        end
    end
end

@testset "CPU Broadcast" begin
    for FT in [Float32, Float64]
        N = 5
        X0 = rand(FT, N, N)
        X1 = rand(FT, N, N) .+ 1000

        for method in [
            SecantMethod(X0, X1),
            RegulaFalsiMethod(X0, X1),
            NewtonsMethodAD(X0),
            NewtonsMethod(X0, f′),
        ]
            sol = RootSolvers.find_zero.(Ref(f), method, CompactSolution())
            converged = map(x -> x.converged, sol)
            X_roots = map(x -> x.root, sol)
            @test all(converged)
            @test eltype(X_roots) == eltype(X0)
            @test all(X_roots .≈ x̃)
        end
    end
end

@testset "CPU isbits Broadcast" begin
    for FT in [Float32, Float64]
        N = 5
        X0 = SArray{Tuple{N, N}, FT}(rand(FT, N, N))
        X1 = SArray{Tuple{N, N}, FT}(rand(FT, N, N)) .+ 1000

        for method in [
            SecantMethod(X0, X1),
            RegulaFalsiMethod(X0, X1),
            NewtonsMethodAD(X0),
            NewtonsMethod(X0, f′),
        ]
            sol = RootSolvers.find_zero.(Ref(f), method, CompactSolution())
            converged = map(x -> x.converged, sol)
            X_roots = map(x -> x.root, sol)
            @test isbits(method)
            @test all(converged)
            @test eltype(X_roots) == eltype(X0)
            @test all(X_roots .≈ x̃)
        end
    end
end

@testset "RootSolvers - compact solution non-converged cases" begin
    F′ = typeof(f′)
    for FT in [Float32, Float64]
        for tol in
            [ResidualTolerance{FT}(1e-6), SolutionTolerance{FT}(1e-1), nothing]
            for method in [
                SecantMethod{FT}(0.0, 1000.0),
                RegulaFalsiMethod{FT}(0.0, 1000.0),
                NewtonsMethodAD{FT}(1.0),
                NewtonsMethod{FT, F′}(1.0, f′),
            ]
                sol = find_zero(f, method, CompactSolution(), tol, 1)
                @test !sol.converged
                @test sol.root isa FT
            end
        end
    end
end

@testset "RootSolvers - verbose solution correctness" begin
    F′ = typeof(f′)
    for FT in [Float32, Float64]
        for method in [
            SecantMethod{FT}(0.0, 1000.0),
            RegulaFalsiMethod{FT}(0.0, 1000.0),
            NewtonsMethodAD{FT}(1.0),
            NewtonsMethod{FT, F′}(1.0, f′),
        ]
            sol = find_zero(f, method, VerboseSolution())
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
end
