using Test
using RootSolvers
using StaticArrays

include("test_helper.jl")

@testset "Convergence reached" begin
    for problem in problem_list
        FT = typeof(problem.x̃)
        for sol_type in [CompactSolution(), VerboseSolution()]
            for tol in get_tolerances(FT)
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper, problem.f′)
                    if problem.x_init isa AbstractArray
                        sol = RootSolvers.find_zero.(Ref(problem.f), method, sol_type, tol)
                        converged = map(x -> x.converged, sol)
                        X_roots = map(x -> x.root, sol)
                        @test isbits(method)
                        @test all(converged)
                        @test eltype(X_roots) == eltype(problem.x_init)
                        @test all(isapprox.(X_roots, problem.x̃, rtol=0.01))
                        test_verbose!(sol_type, sol, problem, tol, all(converged))
                    else
                        sol = find_zero(problem.f, method, sol_type, tol)
                        @test isbits(method)
                        @test sol.converged
                        @test sol.root isa FT
                        @test isapprox(sol.root, problem.x̃, rtol=0.01)
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
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper, problem.f′)
                    if problem.x_init isa AbstractArray
                        sol = RootSolvers.find_zero.(Ref(problem.f), method, sol_type, tol, 1)
                        converged = map(x -> x.converged, sol)
                        X_roots = map(x -> x.root, sol)
                        @test isbits(method)
                        @test !any(converged)
                        @test eltype(X_roots) == eltype(problem.x_init)
                        test_verbose!(sol_type, sol, problem, tol, any(converged))
                    else
                        sol = find_zero(problem.f, method, sol_type, tol, 1)
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
