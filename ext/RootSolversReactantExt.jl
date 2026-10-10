module RootSolversReactantExt

import RootSolvers
import Reactant

# Reactant's scalar wrappers (`TracedRNumber{T}` while tracing, `ConcretePJRTNumber{T,D}`
# / `ConcreteIFRTNumber{T,D}` at the boundary) are `<: Number` but not `<: Real`, and they
# do not unwrap through `eltype` — `eltype(TracedRNumber{Float32}) === TracedRNumber{Float32}`.
# Without this method `base_type` would hand the wrapper itself to `_default_tol_value`,
# which only has methods for plain floating-point types, so `default_tol` (and hence every
# `find_zero` call that does not pass an explicit tolerance) would fail with a `MethodError`
# during tracing. `Reactant.RNumber{T}` is the common supertype of both wrappers.
RootSolvers.base_type(::Type{RN}) where {T, RN <: Reactant.RNumber{T}} =
    RootSolvers.base_type(T)

end
