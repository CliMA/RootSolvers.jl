"""
    RootSolvers

Contains functions for solving roots of non-linear
equations. See [`find_zero`](@ref).


## Example

```julia
julia> using RootSolvers

julia> sol = find_zero(x -> x^2 - 100^2,
                       SecantMethod{Float64}(0.0, 1000.0),
                       CompactSolution());

julia> sol
RootSolvers.CompactSolutionResults{Float64}(99.99999999994358, true)

julia> sol.root
99.99999999994358
```

"""
module RootSolvers

export find_zero,
    SecantMethod, RegulaFalsiMethod, NewtonsMethodAD, NewtonsMethod
export CompactSolution, CompactCachedSolution, VerboseSolution
export AbstractConvergenceCriteria, ResidualTolerance, SolutionTolerance, RelativeSolutionTolerance,
    RelativeOrAbsoluteSolutionTolerance

import ForwardDiff

const maxiters_default = 10_000

abstract type RootSolvingMethod{XT} end
Base.broadcastable(method::RootSolvingMethod) = Ref(method)

"""
    SecantMethod

# Fields
 - `x0` lower bound
 - `x1` upper bound
"""
struct SecantMethod{XT} <: RootSolvingMethod{XT}
    x0::XT
    x1::XT
end

"""
    RegulaFalsiMethod

# Fields
 - `x0` lower bound
 - `x1` upper bound
"""
struct RegulaFalsiMethod{XT} <: RootSolvingMethod{XT}
    x0::XT
    x1::XT
end

"""
    NewtonsMethodAD

# Fields
 - `x0` initial guess
"""
struct NewtonsMethodAD{XT} <: RootSolvingMethod{XT}
    x0::XT
end

"""
    NewtonsMethod

# Fields
 - `x0` initial guess
"""
struct NewtonsMethod{XT} <: RootSolvingMethod{XT}
    x0::XT
end

abstract type SolutionType end
Base.broadcastable(soltype::SolutionType) = Ref(soltype)

"""
    VerboseSolution <: SolutionType

Used to return a [`VerboseSolutionResults`](@ref)
"""
struct VerboseSolution <: SolutionType end

abstract type AbstractSolutionResults{XT} end

"""
    VerboseSolutionResults{XT} <: AbstractSolutionResults{XT}

Result returned from `find_zero` when
`VerboseSolution` is passed as the `soltype`.
"""
struct VerboseSolutionResults{XT, FT} <: AbstractSolutionResults{XT}
    "solution ``x^*`` of the root of the equation ``f(x^*) = 0``"
    root::XT
    "indicates convergence"
    converged::Bool
    "error of the root of the equation ``f(x^*) = 0``"
    err::FT
    "number of iterations performed"
    iter_performed::Int
    "solution per iteration"
    root_history::Vector{XT}
    "error of the root of the equation ``f(x^*) = 0`` per iteration"
    err_history::Vector{XT}
end
SolutionResults(soltype::VerboseSolution, args...) =
    VerboseSolutionResults(args...)

"""
    CompactSolution <: SolutionType

Used to return a [`CompactSolutionResults`](@ref)
"""
struct CompactSolution <: SolutionType end

"""
    CompactCachedSolution <: SolutionType

Used to return a [`CompactCachedSolutionResults`](@ref)
"""
struct CompactCachedSolution <: SolutionType end

"""
    CompactSolutionResults{XT} <: AbstractSolutionResults{XT, FT}

Result returned from `find_zero` when
`CompactSolution` is passed as the `soltype`.

To extract the root, use
```julia
sol = RootSolvers.find_zero(...)
sol.root
```
"""
struct CompactSolutionResults{XT} <: AbstractSolutionResults{XT}
    "solution ``x^*`` of the root of the equation ``f(x^*) = 0``"
    root::XT
    "indicates convergence"
    converged::Bool
end
SolutionResults(soltype::CompactSolution, root, converged, args...) =
    CompactSolutionResults(root, converged)

struct CompactCachedSolutionResults{XT, C} <: AbstractSolutionResults{XT}
    "solution ``x^*`` of the root of the equation ``f(x^*) = 0``"
    root::XT
    "indicates convergence"
    converged::Bool
    "Cache when computing ``f(x)``"
    cache::C
end
SolutionResults(soltype::CompactCachedSolution, root, converged, y, args...) =
    CompactCachedSolutionResults(root, converged, y)

init_history(::VerboseSolution, x::XT) where {XT <: Real} = XT[x]
init_history(::CompactSolution, x) = nothing
init_history(::CompactCachedSolution, x) = nothing
init_history(::VerboseSolution, ::Type{XT}) where {XT} = XT[]
init_history(::CompactSolution, ::Type{XT}) where {XT} = nothing
init_history(::CompactCachedSolution, ::Type{XT}) where {XT} = nothing

function push_history!(
    history::Vector{XT},
    x::XT,
    ::VerboseSolution,
) where {XT <: Real}
    push!(history, x)
end
function push_history!(
    history::Nothing,
    x::XT,
    ::CompactSolution,
) where {XT <: Real}
    nothing
end
function push_history!(
    history::Nothing,
    x::XT,
    ::CompactCachedSolution,
) where {XT <: Real}
    nothing
end


abstract type AbstractConvergenceCriteria{N} end

maxiters(::AbstractConvergenceCriteria{N}) where {N} = N
Base.broadcastable(cc::AbstractConvergenceCriteria) = Ref(cc)

abstract type AbstractRootResult end
struct NumberResult{FX} <: AbstractRootResult
    fx::FX
end
NumberResult(::NumberResult) = x
struct CachedResult{FX, C} <: AbstractRootResult
    fx::FX
    cache::C
end
value(x) = x
value(cr::AbstractRootResult) = cr.fx
cache(x) = x
cache(cr::CachedResult) = cr.cache
NumberResult(x::CachedResult) = x

# struct FunctionAndCache{F, C} <: AbstractRootResult
#     f::F
#     cache::C
# end
# value(x) = x
# value(cr::AbstractRootResult) = cr.fx
# cache(cr::CachedResult) = cr.cache
# NumberResult(x::CachedResult) = x
# (rr::FunctionAndCache)(x) = rr.f(x).value


"""
    SolutionTolerance

A tolerance type based on the solution ``x`` of the equation ``f(x) = 0``
"""
struct SolutionTolerance{N, XT} <: AbstractConvergenceCriteria{N}
    tol::XT
end
SolutionTolerance(maxiters::Int, tol::XT) where {XT} =
    SolutionTolerance{maxiters, XT}(tol)

SolutionTolerance(tol::XT) where {XT} =
    SolutionTolerance(maxiters_default, tol)

"""
    (cc::SolutionTolerance)(x1, x2, y)

Evaluates solution tolerance, based on ``|x2-x1|``
"""
(cc::SolutionTolerance)(x1, x2, y) = abs(x2 - x1) < cc.tol

"""
    ResidualTolerance

A tolerance type based on the residual of the equation ``f(x) = 0``
"""
struct ResidualTolerance{N, FT} <: AbstractConvergenceCriteria{N}
    tol::FT
end
ResidualTolerance(maxiters::Int, tol::FT) where {FT} =
    ResidualTolerance{maxiters, FT}(tol)
ResidualTolerance(tol::FT) where {FT} =
    ResidualTolerance(maxiters_default, tol)

"""
    (cc::ResidualTolerance)(x1, x2, y)

Evaluates residual tolerance, based on ``|f(x)|``
"""
(cc::ResidualTolerance)(x1, x2, y) = abs(y) < cc.tol


"""
    RelativeSolutionTolerance

A tolerance type based on consecutive iterations of solution ``x`` of the equation ``f(x) = 0``
"""
struct RelativeSolutionTolerance{N, XT} <: AbstractConvergenceCriteria{N}
    tol::XT
end
RelativeSolutionTolerance(maxiters::Int, tol::XT) where {XT} =
    RelativeSolutionTolerance{maxiters, XT}(tol)

RelativeSolutionTolerance(tol::XT) where {XT} =
    RelativeSolutionTolerance(maxiters_default, tol)

"""
    (cc::RelativeSolutionTolerance)(x1, x2, y)

Evaluates solution tolerance, based on ``|(x2-x1)/x1|``
"""
(cc::RelativeSolutionTolerance)(x1, x2, y) = abs((x2 - x1)/x1) < cc.tol

"""
    RelativeOrAbsoluteSolutionTolerance(rtol, atol)

A combined tolerance type based on relative and absolute tolerances.

See [`RelativeSolutionTolerance`](@ref) and [`SolutionTolerance`](@ref)
"""
struct RelativeOrAbsoluteSolutionTolerance{N, FT} <: AbstractConvergenceCriteria{N}
    rtol::FT
    atol::FT
end
RelativeOrAbsoluteSolutionTolerance(maxiters::Int, rtol::FT, atol::FT) where {FT} =
    RelativeOrAbsoluteSolutionTolerance{maxiters, FT}(rtol, atol)

RelativeOrAbsoluteSolutionTolerance(rtol::FT, atol::FT) where {FT} =
    RelativeOrAbsoluteSolutionTolerance(maxiters_default, rtol, atol)

"""
    (cc::RelativeOrAbsoluteSolutionTolerance)(x1, x2, y)

Evaluates combined relative and absolute tolerance, based
on ``|(x2-x1)/x1| || |x2-x1|``
"""
(cc::RelativeOrAbsoluteSolutionTolerance)(x1, x2, y) =
    abs((x2 - x1)/x1) < cc.rtol || abs(x2 - x1) < cc.atol

# TODO: CuArrays.jl has trouble with isapprox on 1.1
# we use simple checks for now, will switch to relative checks later.

"""
    sol = find_zero(
            f::F,
            method::RootSolvingMethod{XT},
            soltype::SolutionType,
            convergence_criteria::AbstractConvergenceCriteria = SolutionTolerance{maxiters, XT}(1e-3),
            )

Finds the nearest root of `f`. Returns a the value of the root `x` such
that `f(x) ≈ 0`, and a Boolean value `converged` indicating convergence.

 - `f` function of the equation to find the root
 - `method` can be one of:
    - `SecantMethod()`: [Secant method](https://en.wikipedia.org/wiki/Secant_method)
    - `RegulaFalsiMethod()`: [Regula Falsi method](https://en.wikipedia.org/wiki/False_position_method#The_regula_falsi_(false_position)_method)
    - `NewtonsMethodAD()`: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) using Automatic Differentiation
    - `NewtonsMethod()`: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)
- `soltype` is a solution type which may be one of:
      `CompactSolution` GPU-capable. Solution has `converged` and `root` only, see [`CompactSolutionResults`](@ref)
      `VerboseSolution` CPU-only. Solution has additional diagnostics, see [`VerboseSolutionResults`](@ref)
- `tol` is a tolerance type to determine when to stop iterations.
- `maxiters` is the maximum number of iterations.
"""
function find_zero end

# Main entry point: Dispatch to specific method
function find_zero(
    f::F,
    method::RootSolvingMethod{XT},
    soltype::SolutionType = CompactSolution(),
    convergence_criteria::AbstractConvergenceCriteria = SolutionTolerance{10_000, XT}(1e-3),
) where {XT, F <: Function}
    return find_zero(f, method, method_args(method)..., soltype, convergence_criteria)
end

# Allow broadcast:
function Broadcast.broadcasted(
    ::typeof(find_zero),
    f::F,
    method::RootSolvingMethod{XT},
    soltype::SolutionType,
    convergence_criteria::AbstractConvergenceCriteria = SolutionTolerance{10_000, XT}(1e-3),
) where {XT, F}
    return broadcast(
        find_zero,
        f,
        method,
        method_args(method)...,
        soltype,
        convergence_criteria,
    )
end

####
#### Numerical methods
####

"""
    method_args(method::RootSolvingMethod)

Return tuple of positional args for `RootSolvingMethod`.
"""
function method_args end

method_args(method::SecantMethod) = (method.x0, method.x1)

function find_zero(
    f::F,
    ::SecantMethod,
    x0::XT,
    x1::XT,
    soltype::SolutionType,
    convergence_criteria::AbstractConvergenceCriteria,
) where {F, XT}
    y0 = f(x0)
    y1 = f(x1)
    x_history = init_history(soltype, x0)
    y_history = init_history(soltype, value(y0))
    for i in 1:maxiters(convergence_criteria)
        Δx = x1 - x0
        Δy = value(y1) - value(y0)
        x0, y0 = x1, y1
        push_history!(x_history, x0, soltype)
        push_history!(y_history, value(y0), soltype)
        x1 -= value(y1) * Δx / Δy
        y1 = f(x1)
        if convergence_criteria(x0, x1, value(y1))
            return SolutionResults(
                soltype,
                x1,
                true,
                y1,
                i,
                x_history,
                y_history,
            )
        end
    end
    return SolutionResults(
        soltype,
        x1,
        false,
        y1,
        maxiters(convergence_criteria),
        x_history,
        y_history,
    )
end

method_args(method::RegulaFalsiMethod) = (method.x0, method.x1)

function find_zero(
    f::F,
    ::RegulaFalsiMethod,
    x0::FT,
    x1::FT,
    soltype::SolutionType,
    convergence_criteria::AbstractConvergenceCriteria,
) where {F <: Function, FT}
    y0 = value(f(x0))
    y1 = value(f(x1))
    @assert y0 * y1 < 0
    x_history = init_history(soltype, x0)
    y_history = init_history(soltype, y0)
    lastside = 0
    local x, y
    for i in 1:maxiters(convergence_criteria)
        x = (x0 * y1 - x1 * y0) / (y1 - y0)
        y = f(x)
        push_history!(x_history, x, soltype)
        push_history!(y_history, value(y), soltype)
        if value(y) * y0 < 0
            if convergence_criteria(x, x1, value(y))
                return SolutionResults(
                    soltype,
                    x,
                    true,
                    y,
                    i,
                    x_history,
                    y_history,
                )
            end
            x1, y1 = x, value(y)
            if lastside == +1
                y0 /= 2
            end
            lastside = +1
        else
            if convergence_criteria(x0, x, value(y))
                return SolutionResults(
                    soltype,
                    x,
                    true,
                    y,
                    i,
                    x_history,
                    y_history,
                )
            end
            x0, y0 = x, value(y)
            if lastside == -1
                y1 /= 2
            end
            lastside = -1
        end
    end
    return SolutionResults(soltype, x, false, y, maxiters(convergence_criteria), x_history, y_history)
end


"""
    value_deriv(f, x)

Compute the value and derivative `f(x)` using ForwardDiff.jl.
"""
function value_deriv(f, x::FT) where {FT}
    tag = typeof(ForwardDiff.Tag(f, FT))
    y = f(ForwardDiff.Dual{tag}(x, one(x)))
    v = ForwardDiff.value(tag, y)
    if v isa CachedResult
        fx = ForwardDiff.value(v.fx)
        cr = CachedResult(fx, v.cache)
        return (cr, ForwardDiff.partials(v.fx, 1))
    else
        return (v, ForwardDiff.partials(tag, y, 1))
    end
end

method_args(method::NewtonsMethodAD) = (method.x0,)

function find_zero(
    f::F,
    ::NewtonsMethodAD,
    x0::XT,
    soltype::SolutionType,
    convergence_criteria::AbstractConvergenceCriteria,
) where {F <: Function, XT}
    local y
    x_history = init_history(soltype, XT)
    y_history = init_history(soltype, XT)
    if soltype isa VerboseSolution
        y, y′ = value_deriv(f, x0)
        push_history!(x_history, x0, soltype)
        push_history!(y_history, value(y), soltype)
    end
    for i in 1:maxiters(convergence_criteria)
        y, y′ = value_deriv(f, x0)
        x1 = x0 - value(y) / y′
        push_history!(x_history, x1, soltype)
        push_history!(y_history, value(y), soltype)
        if convergence_criteria(x0, x1, value(y))
            return SolutionResults(
                soltype,
                x1,
                true,
                y,
                i,
                x_history,
                y_history,
            )
        end
        x0 = x1
    end
    return SolutionResults(
        soltype,
        x0,
        false,
        y,
        maxiters(convergence_criteria),
        x_history,
        y_history,
    )
end

method_args(method::NewtonsMethod) = (method.x0,)

function find_zero(
    f::F,
    ::NewtonsMethod,
    x0::FT,
    soltype::SolutionType,
    convergence_criteria::AbstractConvergenceCriteria,
) where {F <: Function, FT}
    x_history = init_history(soltype, FT)
    y_history = init_history(soltype, FT)
    if soltype isa VerboseSolution
        y, y′ = f(x0)
        push_history!(x_history, x0, soltype)
        push_history!(y_history, value(y), soltype)
    end
    for i in 1:maxiters(convergence_criteria)
        y, y′ = f(x0)
        x1 = x0 - value(y) / y′
        push_history!(x_history, x1, soltype)
        push_history!(y_history, value(y), soltype)
        if convergence_criteria(x0, x1, value(y))
            return SolutionResults(
                soltype,
                x1,
                true,
                y,
                i,
                x_history,
                y_history,
            )
        end
        x0 = x1
    end
    return SolutionResults(
        soltype,
        x0,
        false,
        y,
        maxiters(convergence_criteria),
        x_history,
        y_history,
    )
end

end
