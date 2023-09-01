using Test
using RootSolvers
using StaticArrays

struct RootSolvingProblem{S,F,FF′,FT,FTA,N}
  "Name"
  name::S
  "Root equation"
  f::F
  "Root and derivative of f"
  ff′::FF′
  "Exact solution"
  x̃::FT
  "Initial guess, for iterative methods"
  x_init::FTA
  "Lower bound for bracketed methods"
  x_lower::FTA
  "Upper bound for bracketed methods"
  x_upper::FTA
end
function RootSolvingProblem(args...)
  S = typeof(args[1])
  F = typeof(args[2])
  FF′ = typeof(args[3])
  FT = eltype(args[end])
  FTA = typeof(args[end])
  N = length(args[end])
  RootSolvingProblem{S,F,FF′,FT,FTA,N}(args...)
end

problem_size() = 5
problem_size(::RootSolvingProblem{S,F,FF′,FT,FTA,N}) where {S,F,FF′,FT,FTA,N} = N
float_types() = [Float32, Float64]
get_tolerances(FT) = [
  ResidualTolerance{FT}(1e-6),
  SolutionTolerance{FT}(1e-3),
  RelativeSolutionTolerance{FT}(sqrt(eps(FT))),
  RelativeOrAbsoluteSolutionTolerance{FT}(100*eps(FT), 1e-6),
  nothing,
]

test_verbose!(::CompactSolution, sol, problem, tol, converged) = nothing
function test_verbose!(::VerboseSolution, sol, problem, tol, converged)
  # Not sure what else to test here
  sol isa AbstractArray || @test length(sol.root_history) == length(sol.err_history)

  if converged
    if tol isa ResidualTolerance
      if sol isa AbstractArray
        @test all(map(x -> problem.f(x.root), sol) .< tol.tol)
      else
        @test problem.f(sol.root) < tol.tol
      end
    end
  end
end

"""
    expand_data_inputs!(problem_list, N)

Expands data inputs in `problem_list` from scalars to SArray's
"""
function expand_data_inputs!(problem_list, ε, N)
  for problem in deepcopy(problem_list)
    FT = typeof(problem.x̃)
    push!(problem_list,
      RootSolvingProblem(
        problem.name,
        problem.f,
        problem.ff′,
        problem.x̃,
        SArray{Tuple{N, N}, FT}(problem.x_init  .+ FT(ε)*rand(FT, N, N)),
        SArray{Tuple{N, N}, FT}(problem.x_lower .+ FT(ε)*rand(FT, N, N)),
        SArray{Tuple{N, N}, FT}(problem.x_upper .+ FT(ε)*rand(FT, N, N))
        ))
  end
end

#####
##### Construct problem list
#####

"""
    get_methods(x_init, x_lower, x_upper)

Numerical methods to test, given arguments from `RootSolvingProblem`.
"""
function get_methods(x_init, x_lower, x_upper)
    return (
    SecantMethod(x_lower, x_upper),
    RegulaFalsiMethod(x_lower, x_upper),
    NewtonsMethodAD(x_init),
    NewtonsMethod(x_init)
    )
end

# Convenience types for dispatching
# (since instances of SecantMethod
# are not `isbits` with `CuArray`s).
struct SecantMethodType end
struct RegulaFalsiMethodType end
struct NewtonsMethodADType end
struct NewtonsMethodType end

# Convenience methods for unifying interfaces
# in test suite:
get_method(::SecantMethodType, x_init, x_lower, x_upper) = SecantMethod(x_lower, x_upper)
get_method(::RegulaFalsiMethodType, x_init, x_lower, x_upper) = RegulaFalsiMethod(x_lower, x_upper)
get_method(::NewtonsMethodADType, x_init, x_lower, x_upper) = NewtonsMethodAD(x_init)
get_method(::NewtonsMethodType, x_init, x_lower, x_upper) = NewtonsMethod(x_init)

#####
##### Construct problem list
#####

problem_list = RootSolvingProblem[]

for FT in float_types()
  push!(problem_list, RootSolvingProblem(
    "simple quadratic",
    x -> x^2 - 100^2,
    x -> (x^2 - 100^2, 2x),
    FT(100),
    FT(1),
    FT(0),
    FT(1000),
    ))
end

expand_data_inputs!(problem_list, 0.1, problem_size())

