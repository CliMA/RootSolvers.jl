using Printf

@testset "solution pretty printing" begin
    @testset "CompactSolution" begin
        sol = find_zero(x -> x^2 - 100^2,
                       SecantMethod{Float64}(0.0, 1000.0),
                       CompactSolution());
        sol_str = sprint(show, sol)
        @test startswith(sol_str, "CompactSolutionResults{Float64}")
        @test contains(sol_str, "converged")
        sol = find_zero(x -> x^2 - 100^2,
                              SecantMethod{Float64}(0.0, 1e3),
                              CompactSolution(), 
                              RelativeSolutionTolerance(eps(10.0)), 
                              2)
        sol_str = sprint(show, sol)
        @test startswith(sol_str, "CompactSolutionResults{Float64}")
        @test contains(sol_str, "failed to converge")
    end
    @testset "VerboseSolution" begin
        sol = find_zero(x -> x^2 - 100^2,
                       SecantMethod{Float64}(0.0, 1000.0),
                       VerboseSolution());
        sol_str = sprint(show, sol)
        @test startswith(sol_str, "VerboseSolutionResults{Float64}")
        @test contains(sol_str, "converged")
        @test contains(sol_str, "Root: $(sol.root)")
        @test contains(sol_str, "Error: $(sol.err)")
        @test contains(sol_str, "Iterations: $(length(sol.root_history)-1)")
        @test contains(sol_str, "History")
        sol = find_zero(x -> x^2 - 100^2,
                              SecantMethod{Float64}(0.0, 1e3),
                              VerboseSolution(), 
                              RelativeSolutionTolerance(eps(10.0)), 
                              2)
        sol_str = sprint(show, sol)
        @test startswith(sol_str, "VerboseSolutionResults{Float64}")
        @test contains(sol_str, "failed to converge")

    end
end
