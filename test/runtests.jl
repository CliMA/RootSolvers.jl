if get(ARGS, 1, "Array") == "CuArray"
    using CUDA
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
        for sol_type in [CompactSolution(), CompactCachedSolution(), VerboseSolution()]
            for cc in get_convergence_criteria(RootSolvers.maxiters_default, FT)
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper)
                    f = method isa NewtonsMethod ? problem.ff′ : problem.f
                    # method isa NewtonsMethodAD && continue
                    if problem.x_init isa AbstractArray
                        sol = RootSolvers.find_zero.(Ref(f), method, sol_type, cc)
                        converged = map(x -> x.converged, sol)
                        X_roots = map(x -> RootSolvers.value(x.root), sol)
                        @test isbits(method)
                        @test all(converged)
                        @test eltype(X_roots) == eltype(problem.x_init)
                        @test all(X_roots .≈ problem.x̃)
                        test_verbose!(sol_type, sol, problem, cc, all(converged))
                    else
                        sol = find_zero(f, method, sol_type, cc)
                        @test isbits(method)
                        @test sol.converged
                        @test RootSolvers.value(sol.root) isa FT
                        @test RootSolvers.value(sol.root) ≈ problem.x̃
                        test_verbose!(sol_type, sol, problem, cc, sol.converged)
                    end
                end
            end
        end
    end
end

@testset "Convergence reached cached result" begin
    for problem in problem_list
        FT = typeof(problem.x̃)
        for sol_type in [CompactSolution(), VerboseSolution()]
            for cc in get_convergence_criteria(RootSolvers.maxiters_default, FT)
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper)
                    f = method isa NewtonsMethod ? problem.ff′ : problem.f
                    method isa NewtonsMethodAD || continue
                    fc = if method isa NewtonsMethod
                        x -> begin
                            (fx, f′x) = f(x)
                            (RootSolvers.CachedResult(fx, fx+1), f′x)
                        end
                    else
                        x -> begin
                            fx = f(x)
                            RootSolvers.CachedResult(fx, fx+1)
                        end
                    end
                    if problem.x_init isa AbstractArray
                        sol = RootSolvers.find_zero.(Ref(fc), method, sol_type, cc)
                        converged = map(x -> x.converged, sol)
                        X_roots = map(x -> RootSolvers.value(x.root), sol)
                        @test isbits(method)
                        @test all(converged)
                        @test eltype(X_roots) == eltype(problem.x_init)
                        @test all(X_roots .≈ problem.x̃)
                        test_verbose!(sol_type, sol, problem, cc, all(converged))
                    else
                        sol = find_zero(fc, method, sol_type, cc)
                        @test isbits(method)
                        @test sol.converged
                        @test RootSolvers.value(sol.root) isa FT
                        @test RootSolvers.value(sol.root) ≈ problem.x̃
                        test_verbose!(sol_type, sol, problem, cc, sol.converged)
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
            for cc in get_convergence_criteria(1, FT)
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper)
                    f = method isa NewtonsMethod ? problem.ff′ : problem.f
                    if problem.x_init isa AbstractArray
                        sol = RootSolvers.find_zero.(Ref(f), method, sol_type, cc)
                        converged = map(x -> x.converged, sol)
                        X_roots = map(x -> RootSolvers.value(x.root), sol)
                        @test isbits(method)
                        @test !any(converged)
                        @test eltype(X_roots) == eltype(problem.x_init)
                        test_verbose!(sol_type, sol, problem, cc, any(converged))
                    else
                        sol = find_zero(f, method, sol_type, cc)
                        @test isbits(method)
                        @test !sol.converged
                        @test RootSolvers.value(sol.root) isa FT
                        test_verbose!(sol_type, sol, problem, cc, sol.converged)
                    end
                end
            end
        end
    end
end

include("runtests_kernel.jl")
