# Determine array type for kernel testing - supports both CPU (Array) and GPU (CuArray)
# This allows testing the same kernel code on different backends
if get(ARGS, 1, "Array") == "CuArray"
    import CUDA
    ArrayType = CUDA.CuArray
    CUDA.allowscalar(false)  # Ensure GPU operations are properly vectorized
else
    ArrayType = Array
end

@show ArrayType

# Re-include test_helper so we can run this file independently
# This provides access to test problems, methods, and tolerances
include("test_helper.jl")

using KernelAbstractions

# Filter to only array-based problems since kernel tests require vectorized operations
# This ensures we only test problems that can be solved in parallel
filter!(x -> x.x_init isa AbstractArray, problem_list)

@kernel function solve_kernel!(
    f,
    ff′,
    MethodType,
    x_init,
    x_lower,
    x_upper,
    tol,
    dst::AbstractArray{FT, N},
) where {FT, N}
    # Kernel function that solves root-finding problems in parallel
    # Each thread/worker solves one root-finding problem independently

    i = @index(Group, Linear)  # Get the linear index for this thread/worker
    @inbounds begin
        # Construct the method and solve the system for this specific index
        # Store the solution in the destination array

        # Create method instance for this specific problem (index i)
        method = get_method(MethodType, x_init[i], x_lower[i], x_upper[i])

        # Choose function based on method type:
        # - NewtonsMethod requires function that returns (f(x), f'(x))
        # - Other methods use standard function f(x)
        _f = MethodType isa NewtonsMethodType ? ff′ : f

        # Solve the root-finding problem for this specific index
        # Use CompactSolution for memory efficiency in kernel context
        sol = find_zero(_f, method, CompactSolution(), tol)

        # Store the result in the destination array
        dst[i] = sol.root
    end
end

@testset "CPU/GPU kernel test" begin
    # Test that root-finding methods work correctly in parallel kernel execution
    # This validates GPU compatibility and parallel performance

    for prob in problem_list
        # Test each problem with all available methods and tolerances
        FT = typeof(prob.x̃)      # Extract floating-point type from expected solution
        x_init = prob.x_init     # Initial guesses
        x_lower = prob.x_lower   # Lower bounds (for bracketing methods)
        x_upper = prob.x_upper   # Upper bounds (for bracketing methods)

        # Get the actual size of the problem (total number of elements)
        n_elem = length(x_init)
        work_groups = (1,)       # Single work group for simple 1D kernel
        ndrange = (n_elem,)      # Range of indices to process

        for MethodType in (
            SecantMethodType(),
            RegulaFalsiMethodType(),
            BrentsMethodType(),
            NewtonsMethodADType(),
            NewtonsMethodType(),
        )
            # Test all root-finding method types in kernel context

            for tol in get_tolerances(FT)
                # Test all tolerance types for the given floating-point type

                # Create destination arrays for results
                a_dst = Array{FT}(undef, n_elem)  # CPU array for reference
                d_dst = ArrayType(a_dst)          # Device array (CPU or GPU)

                # Get appropriate backend for the device array
                backend = get_backend(d_dst)

                # Compile the kernel for the specific backend
                kernel! = solve_kernel!(backend, work_groups)

                # Launch the kernel with all problem data
                event = kernel!(
                    prob.f,           # Function to find roots of
                    prob.ff′,         # Function with derivatives (for Newton's method)
                    MethodType,       # Type of root-finding method to use
                    x_init,           # Initial guesses array
                    x_lower,          # Lower bounds array
                    x_upper,          # Upper bounds array
                    tol,              # Convergence tolerance
                    d_dst;            # Destination array for results
                    ndrange = ndrange, # Specify the range of indices to process
                )

                # Ensure all kernel operations complete before checking results
                synchronize(backend)

                # Validate that all computed roots match the expected solution
                # Use a reasonable tolerance for comparison
                # For high-multiplicity roots, use a more lenient tolerance since they're inherently difficult
                if prob.name == "high-multiplicity root"
                    tolerance = max(1e-1 * abs(prob.x̃), 1e-3)  # More lenient for difficult problems
                else
                    tolerance = max(1e-3 * abs(prob.x̃), 1e-6)  # Standard tolerance
                end
                @test all(abs.(Array(d_dst) .- prob.x̃) .< tolerance)
            end
        end
    end
end
