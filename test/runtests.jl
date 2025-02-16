if get(ARGS, 1, "Array") == "CuArray"
    import CUDA
    ArrayType = CUDA.CuArray
    CUDA.allowscalar(false)
else
    ArrayType = Array
end

using Test
using RootSolvers
using StaticArrays

include("test_helper.jl")

@testset "Convergence reached" begin
    for problem in problem_list
        FT = typeof(problem.x̃)
        for sol_type in [CompactSolution(), VerboseSolution()]
            for tol in get_tolerances(FT)
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper)
                    f = method isa NewtonsMethod ? problem.ff′ : problem.f
                    if problem.x_init isa AbstractArray
                        sol = RootSolvers.find_zero.(Ref(f), method, sol_type, tol)
                        converged = map(x -> x.converged, sol)
                        X_roots = map(x -> x.root, sol)
                        @test isbits(method)
                        @test all(converged)
                        @test eltype(X_roots) == eltype(problem.x_init)
                        @test all(X_roots .≈ problem.x̃)
                        test_verbose!(sol_type, sol, problem, tol, all(converged))
                    else
                        sol = find_zero(f, method, sol_type, tol)
                        @test isbits(method)
                        @test sol.converged
                        @test sol.root isa FT
                        @test sol.root ≈ problem.x̃
                        test_verbose!(sol_type, sol, problem, tol, sol.converged)
                    end
                end
            end
        end
    end
end

@testset "Convergence not reached" begin
    for problem in problem_list
        FT = typeof(problem.x̃)
        for sol_type in [CompactSolution(), VerboseSolution()]
            for tol in get_tolerances(FT)
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper)
                    f = method isa NewtonsMethod ? problem.ff′ : problem.f
                    if problem.x_init isa AbstractArray
                        sol = RootSolvers.find_zero.(Ref(f), method, sol_type, tol, 1)
                        converged = map(x -> x.converged, sol)
                        X_roots = map(x -> x.root, sol)
                        @test isbits(method)
                        @test !any(converged)
                        @test eltype(X_roots) == eltype(problem.x_init)
                        test_verbose!(sol_type, sol, problem, tol, any(converged))
                    else
                        sol = find_zero(f, method, sol_type, tol, 1)
                        @test isbits(method)
                        @test !sol.converged
                        @test sol.root isa FT
                        test_verbose!(sol_type, sol, problem, tol, sol.converged)
                    end
                end
            end
        end
    end
end

@testset "Check small Δy" begin
    # Test PR: #56
    ## Δy is small and we converged
    sol = find_zero(x -> x^3, SecantMethod{Float64}(1e-8, 1e-8 + 1e-24), VerboseSolution())
    @test sol.converged === true  # Δx is small
    y0, y1 = sol.err_history
    @test abs(y0 - y1) ≤ 2 * eps(Float64)  # Δy is small

    ## Δy is small, but we didn't converge (e.g. found two distinct roots)
    sol = find_zero(x -> x^2 - 1, SecantMethod{Float64}(-1, 1), VerboseSolution())
    @test sol.converged === false  # Δx is large
    y0, y1 = sol.err_history
    @test abs(y0 - y1) ≤ 2 * eps(Float64)  # Δy is small
end

include("runtests_kernel.jl")

include("test_printing.jl")
