push!(LOAD_PATH, joinpath(@__DIR__, "..", "env", "GPU"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "env", "test"))

using Test
using RootSolvers

using StaticArrays
using Adapt
using CUDAdrv
using CUDAnative
using CuArrays

CuArrays.allowscalar(false)
@testset "GPU tests" begin
    f(x) = x^2 - 100^2
    f′(x) = 2x
    x̃ = 100
    for FT in [Float32, Float64]

        # Make isbits array, so that `method` remains isbits
        N = 5
        _X0 = SArray{Tuple{N, N}, FT}(rand(FT, N, N))
        _X1 = SArray{Tuple{N, N}, FT}(rand(FT, N, N) .+ 1000)

        # Move to the GPU
        X0 = adapt(CuArray, _X0)
        X1 = adapt(CuArray, _X1)

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
            @test all(X_roots .≈ 100)
        end
    end
end
