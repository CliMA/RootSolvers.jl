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
```

"""
module RootSolvers

using DocStringExtensions: FIELDS

export find_zero,
    SecantMethod,
    RegulaFalsiMethod,
    NewtonsMethodAD,
    NewtonsMethod,
    BisectionMethod,
    BrentDekker
export CompactSolution, VerboseSolution
export AbstractTolerance, ResidualTolerance, SolutionTolerance

using KernelAbstractions.Extras: @unroll

import ForwardDiff

# Input types
const FTypes = Union{AbstractFloat, AbstractArray}

abstract type RootSolvingMethod{FT <: FTypes} end
Base.broadcastable(method::RootSolvingMethod) = Ref(method)

"""
    SecantMethod

# Fields
$(FIELDS)
"""
struct SecantMethod{FT} <: RootSolvingMethod{FT}
    "lower bound"
    x0::FT
    "upper bound"
    x1::FT
end

"""
    RegulaFalsiMethod

# Fields
$(FIELDS)
"""
struct RegulaFalsiMethod{FT} <: RootSolvingMethod{FT}
    "lower bound"
    x0::FT
    "upper bound"
    x1::FT
end

"""
    NewtonsMethodAD

# Fields
$(FIELDS)
"""
struct NewtonsMethodAD{FT} <: RootSolvingMethod{FT}
    "initial guess"
    x0::FT
end

"""
    NewtonsMethod

# Fields
$(FIELDS)
"""
struct NewtonsMethod{FT, F′ <: Function} <: RootSolvingMethod{FT}
    "initial guess"
    x0::FT
    "`f′` derivative of function `f` whose zero is sought"
    f′::F′
end

"""
    BisectionMethod

# Fields
$(FIELDS)
"""
struct BisectionMethod{FT} <: RootSolvingMethod{FT}
    x0::FT
    x1::FT
end

"""
    BrentDekker

# Fields
$(FIELDS)
"""
struct BrentDekker{FT} <: RootSolvingMethod{FT}
    "lower bound"
    a::FT
    "upper bound"
    b::FT
end

abstract type SolutionType end
Base.broadcastable(soltype::SolutionType) = Ref(soltype)

"""
    VerboseSolution <: SolutionType

Used to return a [`VerboseSolutionResults`](@ref)
"""
struct VerboseSolution <: SolutionType end

abstract type AbstractSolutionResults{AbstractFloat} end

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

"""
    CompactSolution <: SolutionType

Used to return a [`CompactSolutionResults`](@ref)
"""
struct CompactSolution <: SolutionType end

"""
    CompactSolutionResults{FT} <: AbstractSolutionResults{FT}

Result returned from `find_zero` when
`CompactSolution` is passed as the `soltype`.
"""
struct CompactSolutionResults{FT} <: AbstractSolutionResults{FT}
    "solution ``x^*`` of the root of the equation ``f(x^*) = 0``"
    root::FT
    "indicates convergence"
    converged::Bool
end
SolutionResults(soltype::CompactSolution, root, converged, args...) =
    CompactSolutionResults(root, converged)

init_history(::VerboseSolution, x::FT) where {FT <: AbstractFloat} = FT[x]
init_history(::CompactSolution, x) = nothing
init_history(::VerboseSolution, ::Type{FT}) where {FT <: AbstractFloat} = FT[]
init_history(::CompactSolution, ::Type{FT}) where {FT <: AbstractFloat} =
    nothing

function push_history!(
    history::Vector{FT},
    x::FT,
    ::VerboseSolution,
) where {FT <: AbstractFloat}
    push!(history, x)
end
function push_history!(
    history::Nothing,
    x::FT,
    ::CompactSolution,
) where {FT <: AbstractFloat}
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

# TODO: CuArrays.jl has trouble with isapprox on 1.1
# we use simple checks for now, will switch to relative checks later.

"""
    sol = find_zero(
            f::F,
            method::RootSolvingMethod{FT},
            soltype::SolutionType,
            tol::Union{Nothing, AbstractTolerance} = nothing,
            maxiters::Union{Nothing, Int} = 10_000,
            )

Attempts to find the nearest root of `f` to `x0` and `x1`. If `sol.converged ==
true` then `sol.root` contains the value the solver converged to, i.e.,
`f(sol.root) ≈ 0`, otherwise `sol.root` is the value of the final iteration.

 - `f` function of the equation to find the root
 - `soltype` is a solution type which may be one of:
      `CompactSolution` GPU-capable. Solution has `converged` and `root` only, see [`CompactSolutionResults`](@ref)
      `VerboseSolution` CPU-only. Solution has additional diagnostics, see [`VerboseSolutionResults`](@ref)
 - `tol` is a tolerance type to determine when to stop iterations.

 - `method` can be one of:
   - `SecantMethod()`: [Secant method](https://en.wikipedia.org/wiki/Secant_method)
     - Parameters
       - `x0`: initial guess (left side)
       - `x1`: initial guess (right side)
   - `RegulaFalsiMethod()`: [Regula Falsi method](https://en.wikipedia.org/wiki/False_position_method#The_regula_falsi_(false_position)_method).
   - `NewtonsMethodAD()`: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) using Automatic Differentiation
     - Parameters
       - `x0`: initial guess
     - Parameters `x0`
   - `NewtonsMethod()`: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)
     - Parameters
       - `x0`: initial guess
       - `f′`: derivative of function `f` whose zero is sought
   - `BisectionMethod()`: [bisection method](https://en.wikipedia.org/wiki/Bisection_method)
     - Parameters `x0` and `x1` should bracket the root
     - if `xatol === nothing` and `maxiters isa Val` then the bisection iteration
       loop will be unrolled
   - `BrentDekker()`: Brent-Dekker Method as described in Brent (1973). It is a
      combination of bisection, secant, and inverse quadratic interpolation which is
      safeguarded to ensure that the step never leaves a known bracket and that
      sufficient progress is made at each iteration
      - Parameters `x0` and `x1` should bracket the root

The optional arguments:
- `tol` is a `ResidualTolerance` or `SolutionTolerance`
- `maxiters` is the maximum number of iterations.

@Book{Brent1973,
  title={Algorithms for Minimization without Derivatives},
  author={Brent, Richard P},
  year={1973},
  publisher={Prentice-Hall Englewood Cliffs, NJ, USA},
}
"""
function find_zero end

# Main entry point: Dispatch to specific method
function find_zero(
    f::F,
    method::RootSolvingMethod{FT},
    soltype::SolutionType,
    tol::Union{Nothing, AbstractTolerance} = nothing,
    maxiters::Union{Nothing, Int} = 10_000,
) where {FT <: FTypes, F <: Function}
    if tol === nothing
        tol = SolutionTolerance{eltype(FT)}(1e-3)
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
    maxiters::Union{Nothing, Int} = 10_000,
) where {FT <: FTypes, F}
    if tol === nothing
        tol = SolutionTolerance{eltype(FT)}(1e-3)
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
    tol::AbstractTolerance{FT},
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
    tol::AbstractTolerance{FT},
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
    tol::AbstractTolerance{FT},
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

method_args(method::NewtonsMethod) = (method.x0, method.f′)

function find_zero(
    f::F,
    ::NewtonsMethod,
    x0::FT,
    f′::F′,
    soltype::SolutionType,
    tol::AbstractTolerance{FT},
    maxiters::Int,
) where {F <: Function, F′ <: Function, FT, IT <: Int}
    x_history = init_history(soltype, FT)
    y_history = init_history(soltype, FT)
    if soltype isa VerboseSolution
        y, y′ = f(x0), f′(x0)
        push_history!(x_history, x0, soltype)
        push_history!(y_history, y, soltype)
    end
    for i in 1:maxiters
        y, y′ = f(x0), f′(x0)
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

method_args(method::BisectionMethod) = (method.x0, method.x1)

function find_zero(
    f::F,
    ::BisectionMethod,
    x0::FT,
    x1::FT,
    soltype::SolutionType,
    tol::AbstractTolerance{FT},
    maxiters::Int,
) where {F, FT <: AbstractFloat}
    _maxiters = ceil(Int, log2(abs(x1 - x0) / tol.tol))
    y0, y1 = f(x0), f(x1)

    x_history = init_history(soltype, FT)
    push_history!(x_history, x0, soltype)
    push_history!(x_history, x1, soltype)

    y_history = init_history(soltype, FT)
    push_history!(y_history, y0, soltype)
    push_history!(y_history, y1, soltype)

    if y0 * y1 > 0
        return SolutionResults(soltype, x0, false, y0, 0, x_history, y_history)
    end

    for i in 1:_maxiters
        x2 = (x0 + x1) / 2
        y2 = f(x2)
        if y2 * y0 < 0
            x1, y1 = x2, y2
        else
            x0, y0 = x2, y2
        end
        push_history!(x_history, x2, soltype)
        push_history!(y_history, y2, soltype)
    end
    x = (x0 + x1) / 2
    return SolutionResults(
        soltype,
        (x0 + x1) / 2,
        true,
        f(x),
        maxiters,
        x_history,
        y_history,
    )
end

function find_zero(
    f::F,
    ::BisectionMethod,
    x0::FT,
    x1::FT,
    soltype::SolutionType,
    ::Nothing,
    ::Val{maxiters},
) where {maxiters, F, FT <: AbstractFloat}
    y0, y1 = f(x0), f(x1)

    x_history = init_history(soltype, FT)
    push_history!(x_history, x0, soltype)
    push_history!(x_history, x1, soltype)

    y_history = init_history(soltype, FT)
    push_history!(y_history, y0, soltype)
    push_history!(y_history, y1, soltype)

    if y0 * y1 > 0
        return SolutionResults(soltype, x0, false, y0, 0, x_history, y_history)
    end

    @unroll for i in 1:maxiters
        x2 = (x0 + x1) / 2
        y2 = f(x2)
        if y2 * y0 < 0
            x1, y1 = x2, y2
        else
            x0, y0 = x2, y2
        end
        push_history!(x_history, x2, soltype)
        push_history!(y_history, y2, soltype)
    end
    x = (x0 + x1) / 2
    return SolutionResults(
        soltype,
        (x0 + x1) / 2,
        true,
        f(x),
        maxiters,
        x_history,
        y_history,
    )
end

method_args(method::BrentDekker) = (method.a, method.b)
# Based on the zero algorithm of  Richard P.  Brent, Algorithms for Minimization
# without Derivatives, Prentice-Hall, Englewood Cliffs, New Jersey, 1973, 195
# pp. (available online at
# https://maths-people.anu.edu.au/~brent/pub/pub011.html)
function find_zero(
    f::F,
    ::BrentDekker,
    a::FT,
    b::FT,
    soltype::SolutionType,
    _tol::AbstractTolerance{FT},
    maxiters::Int,
) where {F, FT <: AbstractFloat}
    x_history = init_history(soltype, FT)
    y_history = init_history(soltype, FT)

    # Evaluate the function
    fa, fb = f(a), f(b)

    push_history!(y_history, fa, soltype)
    push_history!(x_history, a, soltype)
    push_history!(y_history, fb, soltype)
    push_history!(x_history, b, soltype)

    # Not a bracket
    if fa * fb > 0
        return SolutionResults(soltype, b, false, fb, 1, x_history, y_history)
    end

    # Initialize the other side of the bracket
    c, fc = a, fa

    # below d is the current step and e is previous step
    d = e = b - a

    macheps = eps(FT)

    # solution is in bracket [c, b] and a is the previous b
    for i in 1:maxiters

        # `b` should be best guess (even if that means we don't update value)
        # This also resets `a` (the previous `b`) to be the same as `c
        if abs(fc) < abs(fb)
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        end

        push_history!(y_history, fb, soltype)
        push_history!(x_history, b, soltype)

        # tol is used to make sure that we are properly scaled with respect to
        # the input
        tol = 2macheps * abs(b) + _tol.tol

        # bisection step size
        m = (c - b) / 2

        # If the bracket is small enough or we hit the root
        if (abs(m) < tol || fb == 0)
            return SolutionResults(
                soltype,
                b,
                true,
                fb,
                i,
                x_history,
                y_history,
            )
        end

        # if the step we took last time `e` is smaller than the tolerance OR if
        # our function really increased just do bisection
        if abs(e) < tol || abs(fa) < abs(fb)
            d = e = m
        else
            # do inverse quadratic interpolation if we can
            # otherwise do secant
            # The use of p & q is to make things more floating point stable

            r = fb / fc
            if a ≠ c
                s = fb / fa
                q = fa / fc
                p = s * (2m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            else
                p = 2m * r
                q = 1 - r
            end

            # either p or q has to flip signs, since we want p to be positive
            # below we prefer to flip the sign of q
            p > 0 ? (q = -q) : (p = -p)

            # As long as the step keeps us in the bracket and the step is not too
            # large we accept the secant / quadratic step
            #
            # The first condition ensures that the step p / q cause the solution
            # to be in the interval [b, b + (3/4)*(a-b)], e.g, we do not get too
            # close to a which is the worst of the two sides.
            #
            # The second condition ensures that the step is at least half as big
            # as the previous step, e.g., p / q < e / 2 (otherwise we are not
            # converging quickly so we should just bisect)
            if 2p < 3m * q - abs(tol * q) && p < abs(e * q / 2)
                e, d = d, p / q
            else
                d = e = m
            end
        end

        # Save the last step
        a, fa = b, fb

        # As long as the step isn't too small accept it, otherwise take a `tol`
        # sizes step in the correct direction
        b = b + (abs(d) > tol ? d : (m > 0 ? tol : -tol))
        fb = f(b)

        # if fb and fc have same sign then really a, b bracket the root
        if (fb > 0) == (fc > 0)
            c, fc = a, fa
            d = e = b - a
        end
    end

    return SolutionResults(
        soltype,
        b,
        false,
        fb,
        maxiters,
        x_history,
        y_history,
    )
end

end
