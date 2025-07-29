# RootSolvers.jl

RootSolvers.jl is a Julia package for finding roots of nonlinear equations using robust, efficient, and GPU-capable numerical methods. It provides a simple, unified interface for a variety of classic root-finding algorithms, with flexible convergence criteria and solution reporting.

- [Getting Started](GettingStarted.md): Installation, quick start, and how-to guide
- [API Reference](API.md): Full documentation of all methods and types

## Quick Example
See the [Getting Started](GettingStarted.md) page for more details and examples.

```julia
using Pkg
Pkg.add("RootSolvers")
using RootSolvers

# Find the root of x^2 - 100^2 using the secant method
sol = find_zero(x -> x^2 - 100^2, SecantMethod(0.0, 1000.0))

if sol.converged
    println("Root found: ", sol.root)
else
    println("Root not found")
end
```

## Documentation
- [Getting Started](GettingStarted.md)
- [API Reference](API.md)
