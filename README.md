# RootSolvers.jl

![RootSolvers Logo](docs/src/assets/logo.png)

A high-performance root solver package with GPU support and broadcasting across abstract types

RootSolvers.jl provides robust, efficient numerical methods for finding roots of nonlinear equations. It supports broadcasting across abstract types including ClimaCore fields, GPU arrays, and custom field types, making it ideal for high-performance computing applications in climate modeling, machine learning, and scientific computing.

|||
|---------------------:|:----------------------------------------------|
| **Documentation**    | [![dev][docs-dev-img]][docs-dev-url]          |
| **Code Coverage**    | [![codecov][codecov-img]][codecov-url]        |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/RootSolvers.jl/dev/

[codecov-img]: https://codecov.io/gh/CliMA/RootSolvers.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/RootSolvers.jl

## Features

- **Multiple Root-Finding Methods**: Secant, Regula Falsi, Newton's method with automatic differentiation
- **GPU Support**: Full GPU acceleration with CUDA.jl and other GPU array types
- **Abstract Type Broadcasting**: Works with ClimaCore fields, distributed arrays, and custom field types
- **Flexible Convergence Criteria**: Multiple tolerance types for different applications
- **High-Performance**: Optimized for large-scale parallel processing
- **Climate Modeling Ready**: Designed for use with [ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl)

## Quick Example

```julia
using RootSolvers

# Simple scalar root finding
sol = find_zero(x -> x^2 - 4, SecantMethod(0.0, 3.0))

# Broadcasting with ClimaCore fields
using ClimaCore
x0 = ClimaCore.Fields.Field(rand(100, 100))
x1 = ClimaCore.Fields.Field(rand(100, 100))
f(x) = x.^2 .- 2.0
sol = find_zero(f, SecantMethod(x0, x1), CompactSolution())
```
