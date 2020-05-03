# RootSolvers.jl

A simple GPU-capable root solver package.

## Usage

```@example
using RootSolvers

sol = find_zero(x -> x^2 - 100^2,
                SecantMethod{Float64}(0.0, 1000.0),
                CompactSolution());
```
