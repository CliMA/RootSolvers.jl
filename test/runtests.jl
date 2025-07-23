# Determine array type for testing - supports both CPU (Array) and GPU (CuArray) testing
if get(ARGS, 1, "Array") == "CuArray"
    import CUDA
    ArrayType = CUDA.CuArray
    CUDA.allowscalar(false)  # Ensure GPU operations are properly vectorized
else
    ArrayType = Array
end

using Test
using RootSolvers
using StaticArrays

# Include test helper functions that define test problems, methods, and tolerances
include("test_helper.jl")

# Helper function to check if roots are within tolerance of expected solution
tol_factor = 60      # Factor by which solution tolerance can exceed stopping tolerance
function check_root_tolerance(roots, expected_root, problem, tol)
    if tol isa ResidualTolerance
        # For residual tolerance, check that function values are small
        if roots isa AbstractArray
            @test all(map(x -> abs(problem.f(x)), roots) .< tol.tol)
        else
            @test abs(problem.f(roots)) < tol.tol
        end
    elseif tol isa SolutionTolerance
        # For solution tolerance, check that roots are close to expected
        if roots isa AbstractArray
            @test all(abs.(roots .- expected_root) .< tol_factor * tol.tol)
        else
            @test abs(roots - expected_root) < tol_factor *tol.tol
        end
    elseif tol isa RelativeSolutionTolerance
        # For relative tolerance, check relative difference
        # Avoid division by zero
        if abs(expected_root) < eps(typeof(expected_root))
            if roots isa AbstractArray
                @test all(abs.(roots .- expected_root)) .< 50*tol.tol
            else
                @test abs(roots - expected_root) < tol_factor * tol.tol
            end
        else
            if roots isa AbstractArray
                @test all(abs.((roots .- expected_root) ./ expected_root) .< tol_factor * tol.tol)
            else
                @test abs((roots - expected_root) / expected_root) < tol_factor * tol.tol
            end
        end
    elseif tol isa RelativeOrAbsoluteSolutionTolerance
        # For combined tolerance, check both relative and absolute
        if roots isa AbstractArray
            @test all((abs.((roots .- expected_root) ./ expected_root) .< tol_factor * tol.rtol) .| (abs.(roots .- expected_root) .< tol_factor * tol.atol))
        else
            @test (abs((roots - expected_root) / expected_root) < tol_factor * tol.rtol) || (abs(roots - expected_root) < tol_factor * tol.atol)
        end
    else
        # Default tolerance check - use a reasonable default based on floating point precision
        FT = typeof(expected_root)
        default_tol = tol_factor * FT(1e-4)  
        if roots isa AbstractArray
            @test all(abs.(roots .- expected_root) .< default_tol)
        else
            @test abs(roots - expected_root) < default_tol
        end
    end
end

# Helper function to get maxiters based on problem name
function get_maxiters(problem_name)
    if problem_name in ["high-multiplicity root", "steep exponential function", "trigonometric function"]
        return 100_000
    else
        return 10_000
    end
end

# Helper function to run solver and extract results
function run_solver_test(f, method, sol_type, tol, maxiters, is_array)
    if is_array
        sol = RootSolvers.find_zero.(Ref(f), method, sol_type, tol, maxiters)
        converged = map(x -> x.converged, sol)
        roots = map(x -> x.root, sol)
        return sol, converged, roots
    else
        sol = find_zero(f, method, sol_type, tol, maxiters)
        return sol, sol.converged, sol.root
    end
end

@testset "Convergence reached" begin
    # Test that all root-finding methods converge to the correct solution
    # This is the main test suite that validates the core functionality
    for problem in problem_list
        FT = typeof(problem.x̃)  # Extract the floating-point type from the expected solution
        for sol_type in [CompactSolution(), VerboseSolution()]
            # Test both solution types: compact (GPU-friendly) and verbose (with history)
            for tol in get_tolerances(FT)
                # Test all tolerance types for the given floating-point type
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper) 
                #for method in [SecantMethod(problem.x_lower, problem.x_upper)]
                #for method in [NewtonsMethod(problem.x_init)]
                #for method in [NewtonsMethodAD(problem.x_init)]
                #for method in [RegulaFalsiMethod(problem.x_lower, problem.x_upper)]
                    # Test all applicable root-finding methods for this problem
                    
                    # Choose function based on method type:
                    # - NewtonsMethod requires function that returns (f(x), f'(x))
                    # - Other methods use standard function f(x)
                    f = method isa NewtonsMethod ? problem.ff′ : problem.f
                    
                    # Run solver test
                    is_array = problem.x_init isa AbstractArray
                    maxiters = get_maxiters(problem.name)
                    sol, converged, roots = run_solver_test(f, method, sol_type, tol, maxiters, is_array)
                    
                    # Validate results
                    @test isbits(method)                    # Ensure method is bitstype (GPU-compatible)
                    if is_array
                        @test all(converged)                # All should converge
                        @test eltype(roots) == eltype(problem.x_init)  # Type consistency
                        test_verbose!(sol_type, sol, problem, tol, all(converged))  # Additional verbose checks
                    else
                        @test converged                     # Should converge
                        @test roots isa FT                  # Type consistency
                        test_verbose!(sol_type, sol, problem, tol, converged)  # Additional verbose checks
                    end
                    
                    # Check that roots are within a reasonable tolerance of the expected solution
                    check_root_tolerance(roots, problem.x̃, problem, tol)
                end
            end
        end
    end
end

@testset "Convergence not reached" begin
    # Test that methods properly handle cases where convergence is not possible
    # This validates error handling and non-convergence detection
    for problem in problem_list
        FT = typeof(problem.x̃)
        for sol_type in [CompactSolution(), VerboseSolution()]
            for tol in get_tolerances(FT)
                for method in get_methods(problem.x_init, problem.x_lower, problem.x_upper)
                    # Same function selection logic as above
                    f = method isa NewtonsMethod ? problem.ff′ : problem.f
                    
                    # Run solver test with maxiters=1 to force non-convergence
                    is_array = problem.x_init isa AbstractArray
                    sol, converged, roots = run_solver_test(f, method, sol_type, tol, 1, is_array)
                    
                    # Validate non-convergence
                    @test isbits(method)
                    if is_array
                        @test !any(converged)                  # None should converge (maxiters=1)
                        @test eltype(roots) == eltype(problem.x_init)
                        test_verbose!(sol_type, sol, problem, tol, any(converged))
                    else
                        @test !converged                       # Should not converge (maxiters=1)
                        @test roots isa FT
                        test_verbose!(sol_type, sol, problem, tol, converged)
                    end
                end
            end
        end
    end
end

@testset "Check small Δy" begin
    # Test edge case where function values are very close (small Δy)
    # This tests the robustness of the secant method implementation
    
    ## Case 1: Δy is small and we converged
    # This tests the case where function values are nearly identical but we still converge
    # due to small Δx (solution is very close)
    sol = find_zero(x -> x^3, SecantMethod{Float64}(1e-8, 1e-8 + 1e-24), VerboseSolution())
    @test sol.converged === true  # Δx is small
    y0, y1 = sol.err_history
    @test abs(y0 - y1) ≤ 2 * eps(Float64)  # Δy is small

    ## Case 2: Δy is small, but we didn't converge
    # This tests the case where function values are nearly identical but we don't converge
    # because the solution is not close (e.g., found two distinct roots)
    sol = find_zero(x -> x^2 - 1, SecantMethod{Float64}(-1, 1), VerboseSolution())
    @test sol.converged === false  # Should not converge because Δx is large
    y0, y1 = sol.err_history
    @test abs(y0 - y1) ≤ 2 * eps(Float64)  # Verify Δy is small
end

# Include additional test files for specialized functionality
include("runtests_kernel.jl")  # GPU kernel tests using KernelAbstractions
include("test_printing.jl")    # Tests for solution pretty printing
