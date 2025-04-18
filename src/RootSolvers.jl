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
CompactSolutionResults{Float64}:
├── Status: converged
└── Root: 99.99999999994358

julia> sol.root
99.99999999994358
```

"""
module RootSolvers

export find_zero,
    SecantMethod, RegulaFalsiMethod, NewtonsMethodAD, NewtonsMethod
export CompactSolution, VerboseSolution
export AbstractTolerance, ResidualTolerance, SolutionTolerance, RelativeSolutionTolerance,
    RelativeOrAbsoluteSolutionTolerance

import ForwardDiff
import Printf: @printf

base_type(::Type{FT}) where {FT} = FT
base_type(::Type{FT}) where {T, FT <: ForwardDiff.Dual{<:Any, T}} = base_type(T)
base_type(::Type{FT}) where {T, FT <: AbstractArray{T}} = base_type(T)

# Input types
const FTypes = Union{Real, AbstractArray}

abstract type RootSolvingMethod{FT <: FTypes} end
Base.broadcastable(method::RootSolvingMethod) = Ref(method)

"""
    SecantMethod

# Fields
 - `x0` lower bound
 - `x1` upper bound
"""
struct SecantMethod{FT} <: RootSolvingMethod{FT}
    x0::FT
    x1::FT
end

"""
    RegulaFalsiMethod

# Fields
 - `x0` lower bound
 - `x1` upper bound
"""
struct RegulaFalsiMethod{FT} <: RootSolvingMethod{FT}
    x0::FT
    x1::FT
end

"""
    NewtonsMethodAD

# Fields
 - `x0` initial guess
"""
struct NewtonsMethodAD{FT} <: RootSolvingMethod{FT}
    x0::FT
end

"""
    NewtonsMethod

# Fields
 - `x0` initial guess
"""
struct NewtonsMethod{FT} <: RootSolvingMethod{FT}
    x0::FT
end

abstract type SolutionType end
Base.broadcastable(soltype::SolutionType) = Ref(soltype)

"""
    VerboseSolution <: SolutionType

Used to return a [`VerboseSolutionResults`](@ref)
"""
struct VerboseSolution <: SolutionType end

abstract type AbstractSolutionResults{Real} end

"""
    VerboseSolutionResults{FT} <: AbstractSolutionResults{FT}

Result returned from `find_zero` when
`VerboseSolution` is passed as the `soltype`.
"""
struct VerboseSolutionResults{FT} <: AbstractSolutionResults{FT}
    "solution ``x^*`` of the root of the equation ``f(x^*) = 0``"
    root::FT
    "indicates convergence"
    converged::Bool
    "error of the root of the equation ``f(x^*) = 0``"
    err::FT
    "number of iterations performed"
    iter_performed::Int
    "solution per iteration"
    root_history::Vector{FT}
    "error of the root of the equation ``f(x^*) = 0`` per iteration"
    err_history::Vector{FT}
end
SolutionResults(soltype::VerboseSolution, args...) =
    VerboseSolutionResults(args...)

function Base.show(io::IO, sol::VerboseSolutionResults{FT}) where {FT}
    status = sol.converged ? "\e[32mconverged\e[0m" : "\e[31mfailed to converge\e[0m"
    println(io, "VerboseSolutionResults{$FT}:")
    println(io, "├── Status: ", status)
    println(io, "├── Root: ", sol.root)
    println(io, "├── Error: ", sol.err)
    println(io, "├── Iterations: ", sol.iter_performed)
    println(io, "└── History:")
    n_iters = length(sol.root_history)
    for i in 1:n_iters
        if n_iters > 20 && 9 < i < n_iters - 9
            i == 11 && println(io, "    ⋮            ⋮                ⋮")
            i == 12 && println(io, "    ⋮            ⋮                ⋮")
            continue
        end
        prefix = i == n_iters ? "    └──" : "    ├──"
        @printf(io, "%s iter %2d: x = %8.5g, err = %.4g\n", 
            prefix, i, sol.root_history[i], sol.err_history[i])
    end
end

"""
    CompactSolution <: SolutionType

Used to return a [`CompactSolutionResults`](@ref)
"""
struct CompactSolution <: SolutionType end

"""
    CompactSolutionResults{FT} <: AbstractSolutionResults{FT}

Result returned from `find_zero` when
`CompactSolution` is passed as the `soltype`.

To extract the root, use
```julia
sol = RootSolvers.find_zero(...)
sol.root
```
"""
struct CompactSolutionResults{FT} <: AbstractSolutionResults{FT}
    "solution ``x^*`` of the root of the equation ``f(x^*) = 0``"
    root::FT
    "indicates convergence"
    converged::Bool
end
SolutionResults(soltype::CompactSolution, root, converged, args...) =
    CompactSolutionResults(root, converged)

function Base.show(io::IO, sol::CompactSolutionResults{FT}) where {FT}
    status = sol.converged ? "\e[32mconverged\e[0m" : "\e[31mfailed to converge\e[0m"
    println(io, "CompactSolutionResults{$FT}:")
    println(io, "├── Status: ", status)
    println(io, "└── Root: ", sol.root)
end

init_history(::VerboseSolution, x::FT) where {FT <: Real} = FT[x]
init_history(::CompactSolution, x) = nothing
init_history(::VerboseSolution, ::Type{FT}) where {FT <: Real} = FT[]
init_history(::CompactSolution, ::Type{FT}) where {FT <: Real} =
    nothing

function push_history!(
    history::Vector{FT},
    x::FT,
    ::VerboseSolution,
) where {FT <: Real}
    push!(history, x)
end
function push_history!(
    history::Nothing,
    x::FT,
    ::CompactSolution,
) where {FT <: Real}
    nothing
end

abstract type AbstractTolerance{FT <: FTypes} end
Base.broadcastable(tol::AbstractTolerance) = Ref(tol)

"""
    ResidualTolerance

A tolerance type based on the residual of the equation ``f(x) = 0``
"""
struct ResidualTolerance{FT} <: AbstractTolerance{FT}
    tol::FT
end

"""
    (tol::ResidualTolerance)(x1, x2, y)

Evaluates residual tolerance, based on ``|f(x)|``
"""
(tol::ResidualTolerance)(x1, x2, y) = abs(y) < tol.tol

"""
    SolutionTolerance

A tolerance type based on the solution ``x`` of the equation ``f(x) = 0``
"""
struct SolutionTolerance{FT} <: AbstractTolerance{FT}
    tol::FT
end

"""
    (tol::SolutionTolerance)(x1, x2, y)

Evaluates solution tolerance, based on ``|x2-x1|``
"""
(tol::SolutionTolerance)(x1, x2, y) = abs(x2 - x1) < tol.tol

"""
    RelativeSolutionTolerance

A tolerance type based on consecutive iterations of solution ``x`` of the equation ``f(x) = 0``
"""
struct RelativeSolutionTolerance{FT} <: AbstractTolerance{FT}
    tol::FT
end

"""
    (tol::RelativeSolutionTolerance)(x1, x2, y)

Evaluates solution tolerance, based on ``|(x2-x1)/x1|``
"""
(tol::RelativeSolutionTolerance)(x1, x2, y) = abs((x2 - x1)/x1) < tol.tol

"""
    RelativeOrAbsoluteSolutionTolerance(rtol, atol)

A combined tolerance type based on relative and absolute tolerances.

See [`RelativeSolutionTolerance`](@ref) and [`SolutionTolerance`](@ref)
"""
struct RelativeOrAbsoluteSolutionTolerance{FT} <: AbstractTolerance{FT}
    rtol::FT
    atol::FT
end

"""
    (tol::RelativeOrAbsoluteSolutionTolerance)(x1, x2, y)

Evaluates combined relative and absolute tolerance, based
on ``|(x2-x1)/x1| || |x2-x1|``
"""
(tol::RelativeOrAbsoluteSolutionTolerance)(x1, x2, y) =
    abs((x2 - x1)/x1) < tol.rtol || abs(x2 - x1) < tol.atol

# TODO: CuArrays.jl has trouble with isapprox on 1.1
# we use simple checks for now, will switch to relative checks later.

"""
    sol = find_zero(
            f::F,
            method::RootSolvingMethod{FT},
            soltype::SolutionType,
            tol::Union{Nothing, AbstractTolerance} = nothing,
            maxiters::Int = 10_000,
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
    method::RootSolvingMethod{FT},
    soltype::SolutionType = CompactSolution(),
    tol::Union{Nothing, AbstractTolerance} = nothing,
    maxiters::Int = 10_000,
) where {FT <: FTypes, F <: Function}
    if tol === nothing
        tol = SolutionTolerance{base_type(FT)}(1e-3)
    end
    return find_zero(f, method, method_args(method)..., soltype, tol, maxiters)
end

# Allow broadcast:
function Broadcast.broadcasted(
    ::typeof(find_zero),
    f::F,
    method::RootSolvingMethod{FT},
    soltype::SolutionType,
    tol::Union{Nothing, AbstractTolerance} = nothing,
    maxiters::Int = 10_000,
) where {FT <: FTypes, F}
    if tol === nothing
        tol = SolutionTolerance{base_type(FT)}(1e-3)
    end
    return broadcast(
        find_zero,
        f,
        method,
        method_args(method)...,
        soltype,
        tol,
        maxiters,
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
    x0::FT,
    x1::FT,
    soltype::SolutionType,
    tol::AbstractTolerance,
    maxiters::Int,
) where {F <: Function, FT <: FTypes}
    y0 = f(x0)
    y1 = f(x1)
    x_history = init_history(soltype, x0)
    y_history = init_history(soltype, y0)
    for i in 1:maxiters
        Δx = x1 - x0
        Δy = y1 - y0
        x0, y0 = x1, y1
        push_history!(x_history, x0, soltype)
        push_history!(y_history, y0, soltype)
        if abs(Δy) ≤ 2eps(y0)  # catch for division by zero (machine-small Δy)
            return SolutionResults(
                soltype,
                x0,
                abs(Δx) ≤ 2eps(x0), # only declare convergence if Δx is small
                y0,
                i,
                x_history,
                y_history,
            )
        end
        x1 -= y1 * Δx / Δy
        y1 = f(x1)
        if tol(x0, x1, y1)
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
        maxiters,
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
    tol::AbstractTolerance,
    maxiters::Int,
) where {F <: Function, FT}
    y0 = f(x0)
    y1 = f(x1)
    @assert y0 * y1 < 0
    x_history = init_history(soltype, x0)
    y_history = init_history(soltype, y0)
    lastside = 0
    local x, y
    for i in 1:maxiters
        x = (x0 * y1 - x1 * y0) / (y1 - y0)
        y = f(x)
        push_history!(x_history, x, soltype)
        push_history!(y_history, y, soltype)
        if y * y0 < 0
            if tol(x, x1, y)
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
            x1, y1 = x, y
            if lastside == +1
                y0 /= 2
            end
            lastside = +1
        else
            if tol(x0, x, y)
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
            x0, y0 = x, y
            if lastside == -1
                y1 /= 2
            end
            lastside = -1
        end
    end
    return SolutionResults(soltype, x, false, y, maxiters, x_history, y_history)
end


"""
    value_deriv(f, x)

Compute the value and derivative `f(x)` using ForwardDiff.jl.
"""
function value_deriv(f, x::FT) where {FT}
    tag = typeof(ForwardDiff.Tag(f, FT))
    y = f(ForwardDiff.Dual{tag}(x, one(x)))
    ForwardDiff.value(tag, y), ForwardDiff.partials(tag, y, 1)
end

method_args(method::NewtonsMethodAD) = (method.x0,)

function find_zero(
    f::F,
    ::NewtonsMethodAD,
    x0::FT,
    soltype::SolutionType,
    tol::AbstractTolerance,
    maxiters::Int,
) where {F <: Function, FT}
    local y
    x_history = init_history(soltype, FT)
    y_history = init_history(soltype, FT)
    if soltype isa VerboseSolution
        y, y′ = value_deriv(f, x0)
        push_history!(x_history, x0, soltype)
        push_history!(y_history, y, soltype)
    end
    for i in 1:maxiters
        y, y′ = value_deriv(f, x0)
        x1 = x0 - y / y′
        push_history!(x_history, x1, soltype)
        push_history!(y_history, y, soltype)
        if tol(x0, x1, y)
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
        maxiters,
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
    tol::AbstractTolerance,
    maxiters::Int,
) where {F <: Function, FT}
    x_history = init_history(soltype, FT)
    y_history = init_history(soltype, FT)
    if soltype isa VerboseSolution
        y, y′ = f(x0)
        push_history!(x_history, x0, soltype)
        push_history!(y_history, y, soltype)
    end
    for i in 1:maxiters
        y, y′ = f(x0)
        x1 = x0 - y / y′
        push_history!(x_history, x1, soltype)
        push_history!(y_history, y, soltype)
        if tol(x0, x1, y)
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
        maxiters,
        x_history,
        y_history,
    )
end

end
