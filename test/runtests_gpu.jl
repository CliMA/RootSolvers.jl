push!(LOAD_PATH, joinpath(@__DIR__, "..", "env", "GPU"))

using Test
using RootSolvers

using CUDAdrv
using CUDAnative
using CuArrays

CuArrays.allowscalar(false)

@testset "GPU tests" begin
    for FT in [Float32, Float64]
        X0 = cu(rand(FT, 5,5))
        X1 = cu(rand(FT, 5,5)) .+ 1000
        f(x) = x^2 - 100^2

        sol = RootSolvers.find_zero.(f, X0, X1, SecantMethod(), CompactSolution())
        converged = map(x->x.converged, sol)
        X_roots = map(x->x.root, sol)
        @test all(converged)
        @test eltype(X_roots) == eltype(X0)
        @test all(X_roots .≈ 100)
    end
end
