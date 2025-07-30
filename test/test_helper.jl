using Test
using RootSolvers
using StaticArrays

# Define a structured type to represent root-finding test problems
# This encapsulates all the data needed to test a root-finding method
struct RootSolvingProblem{S,F,FF′,FT,FTA,N}
  "Name"                    # Human-readable name for the problem
  name::S
  "Root equation"           # Function f(x) for which we want to find f(x) = 0
  f::F
  "Root and derivative of f"  # Function returning (f(x), f'(x)) for Newton's method
  ff′::FF′
  "Exact solution"          # Known exact solution x̃ where f(x̃) = 0
  x̃::FT
  "Initial guess, for iterative methods"  # Starting point for iterative methods
  x_init::FTA
  "Lower bound for bracketed methods"     # Lower bound for bracketing methods (Regula Falsi)
  x_lower::FTA
  "Upper bound for bracketed methods"     # Upper bound for bracketing methods (Regula Falsi)
  x_upper::FTA
end

# Constructor that automatically infers types from the arguments
# This makes it easier to create test problems without specifying all type parameters
function RootSolvingProblem(args...)
  S = typeof(args[1])      # String type for name
  F = typeof(args[2])      # Function type for f
  FF′ = typeof(args[3])    # Function type for ff′
  FT = eltype(args[end])    # Element type of the last argument (x_upper)
  FTA = typeof(args[end])   # Type of the last argument (x_upper)
  N = length(args[end])     # Length of the last argument (for array problems)
  RootSolvingProblem{S,F,FF′,FT,FTA,N}(args...)
end

# Helper functions for test configuration
problem_size() = 5  # Default size for array-based test problems
problem_size(::RootSolvingProblem{S,F,FF′,FT,FTA,N}) where {S,F,FF′,FT,FTA,N} = N  # Extract size from problem
float_types() = [Float32, Float64]  # Supported floating-point types for testing

# Define all tolerance types to test for each floating-point type
# This ensures comprehensive testing of convergence criteria
get_tolerances(FT) = [
  ResidualTolerance{FT}(1e-4),                           # Based on |f(x)|
  SolutionTolerance{FT}(1e-3),                           # Based on |x_{n+1} - x_n|
  RelativeSolutionTolerance{FT}(1e-3),                   # Based on |(x_{n+1} - x_n)/x_n|
  RelativeOrAbsoluteSolutionTolerance{FT}(1e-3, 1e-3),   # Combined relative/absolute
  nothing,                                               # Default tolerance
]

# Test function for verbose solutions - validates additional properties
# For CompactSolution, no additional tests are needed (memory efficient)
test_verbose!(::CompactSolution, sol, problem, tol, converged) = nothing

# For VerboseSolution, test additional properties like history consistency
function test_verbose!(::VerboseSolution, sol, problem, tol, converged)
    FT = eltype(problem.x_init)
    # Validate that root and error history have the same length
    # (only for scalar solutions, not array solutions)
    if sol isa AbstractArray
        @test all(map(s -> length(s.root_history) == length(s.err_history), sol))
    else
        @test length(sol.root_history) == length(sol.err_history)
    end

    # If converged, test that the residual tolerance is satisfied
    if converged
        if tol isa ResidualTolerance
            if sol isa AbstractArray
                # For array solutions, check all residuals are below tolerance
                @test all(map(x -> abs(problem.f(x.root)) < tol.tol, sol))
            else
                # For scalar solutions, check single residual is below tolerance
                @test abs(problem.f(sol.root)) < tol.tol
            end
        end
    end
end

"""
    expand_data_inputs!(problem_list, N)

Expands data inputs in `problem_list` from scalars to SArray's
This creates array-based versions of scalar test problems for testing broadcasting
"""
function expand_data_inputs!(problem_list, ε, N)
  # Create array-based versions of each scalar problem
  for problem in deepcopy(problem_list)
    FT = typeof(problem.x̃)
    # Add small random perturbations to create array problems
    # This tests robustness to slight variations in initial conditions
    push!(problem_list,
      RootSolvingProblem(
        problem.name,
        problem.f,
        problem.ff′,
        problem.x̃,
        SArray{Tuple{N, N}, FT}(problem.x_init  .+ FT(ε)*rand(FT, N, N)),  # Initial guesses with noise
        SArray{Tuple{N, N}, FT}(problem.x_lower .+ FT(ε)*rand(FT, N, N)),  # Lower bounds with noise
        SArray{Tuple{N, N}, FT}(problem.x_upper .+ FT(ε)*rand(FT, N, N))   # Upper bounds with noise
        ))
  end
end

#####
##### Construct problem list
#####

"""
    get_methods(x_init, x_lower, x_upper)

Numerical methods to test, given arguments from `RootSolvingProblem`.
Returns a tuple of all applicable root-finding methods for the given problem.
"""
function get_methods(x_init, x_lower, x_upper)
    return (
    SecantMethod(x_lower, x_upper),      # Two-point method using linear interpolation
    RegulaFalsiMethod(x_lower, x_upper), # Bracketing method with guaranteed convergence
    BrentsMethod(x_lower, x_upper),      # Brent's method with superlinear convergence
    NewtonsMethodAD(x_init),             # Newton's method with automatic differentiation
    NewtonsMethod(x_init)                # Newton's method with user-provided derivative
    )
end

# Convenience types for dispatching in kernel tests
# These are used because instances of root-finding methods
# are not `isbits` with `CuArray`s, but types are.
struct SecantMethodType end
struct RegulaFalsiMethodType end
struct BrentsMethodType end
struct NewtonsMethodADType end
struct NewtonsMethodType end

# Convenience methods for unifying interfaces in the test suite:
# These allow the same kernel code to work with different method types
get_method(::SecantMethodType, x_init, x_lower, x_upper) = SecantMethod(x_lower, x_upper)
get_method(::RegulaFalsiMethodType, x_init, x_lower, x_upper) = RegulaFalsiMethod(x_lower, x_upper)
get_method(::BrentsMethodType, x_init, x_lower, x_upper) = BrentsMethod(x_lower, x_upper)
get_method(::NewtonsMethodADType, x_init, x_lower, x_upper) = NewtonsMethodAD(x_init)
get_method(::NewtonsMethodType, x_init, x_lower, x_upper) = NewtonsMethod(x_init)

#####
##### Construct problem list
#####

# Initialize the list of test problems
problem_list = RootSolvingProblem[]

# Create test problems for each floating-point type
for FT in float_types()
  # Simple quadratic problem: f(x) = x^2 - 100^2, solution x = 100
  # This tests basic functionality with a well-behaved function
  push!(problem_list, RootSolvingProblem(
    "simple quadratic",
    x -> x^2 - 100^2,                    # Function f(x) = x^2 - 100^2
    x -> (x^2 - 100^2, 2x),              # Function and derivative (f(x), f'(x))
    FT(100),                             # Exact solution x = +/- 100
    FT(1),                               # Initial guess x₀ = 1
    FT(0),                               # Lower bound for bracketing
    FT(1000),                            # Upper bound for bracketing
    ))

  # High-multiplicity root: f(x) = (x-2)^5, solution x = 2
  # This tests methods on functions with zero derivative at the root
  push!(problem_list, RootSolvingProblem(
    "high-multiplicity root",
    x -> (x - 2)^5,                      # Function f(x) = (x-2)^5
    x -> ((x - 2)^5, 5 * (x - 2)^4),     # Function and derivative
    FT(2),                               # Exact solution x = 2
    FT(1),                               # Initial guess x₀ = 1
    FT(0),                               # Lower bound for bracketing
    FT(3),                               # Upper bound for bracketing
    ))

  # Steep exponential function: f(x) = exp(5*(x-1)) - 1, solution x = 1
  # This tests methods on functions with large gradients
  push!(problem_list, RootSolvingProblem(
    "steep exponential function",
    x -> exp(5 * (x - 1)) - 1,          # Function f(x) = exp(5*(x-1)) - 1
    x -> (exp(5 * (x - 1)) - 1, 5 * exp(5 * (x - 1))), # Function and derivative
    FT(1),                               # Exact solution x = 1
    FT(0.5),                             # Initial guess x₀ = 0.5
    FT(0.5),                             # Lower bound for bracketing
    FT(1.5),                             # Upper bound for bracketing
    ))

  # Trigonometric function: f(x) = sin(x), solution x = π
  # This tests finding a specific root when multiple roots exist
  push!(problem_list, RootSolvingProblem(
    "trigonometric function",
    x -> sin(x),                         # Function f(x) = sin(x)
    x -> (sin(x), cos(x)),               # Function and derivative
    FT(π),                               # Exact solution x = π
    FT(3),                               # Initial guess x₀ = 3
    FT(2),                               # Lower bound for bracketing
    FT(4),                               # Upper bound for bracketing
     ))
end

# Expand scalar problems to array problems for testing broadcasting
# This creates array-based versions with small random perturbations
expand_data_inputs!(problem_list, 0.1, problem_size())

