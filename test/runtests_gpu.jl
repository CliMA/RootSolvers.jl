using Test
using RootSolvers

using StaticArrays
using Adapt
using CUDAdrv
using CUDAnative
using CuArrays

include("test_helper.jl")

CuArrays.allowscalar(false)

filter!(x->x.x_init isa AbstractArray, problem_list) # only grab array problems

@testset "GPU tests" begin
    for problem in problem_list
        FT = typeof(problem.x̃)
        for tol in get_tolerances(FT)
            x_init = adapt(CuArray, problem.x_init)
            x_lower = adapt(CuArray, problem.x_lower)
            x_upper = adapt(CuArray, problem.x_upper)
            for method in get_methods(x_init, x_lower, x_upper, problem.f′)
                sol = RootSolvers.find_zero.(Ref(problem.f), method, CompactSolution(), tol)
                converged = map(x -> x.converged, sol)
                X_roots = map(x -> x.root, sol)
                @test isbits(method)
                @test all(converged)
                @test eltype(X_roots) == eltype(x_init)
                @test all(X_roots .≈ problem.x̃)
            end
        end
    end
end
