# Getting Started

RootSolvers.jl is a Julia package for finding roots of nonlinear equations using robust, efficient, and GPU-capable numerical methods. It provides a simple, unified interface for a variety of classic root-finding algorithms, with flexible convergence criteria and solution reporting.

---

## Installation

The package is registered in the Julia General registry.

**Requirements:**
- Julia 1.8 or newer

**Stable Release:**
```julia
using Pkg
Pkg.add("RootSolvers")
```

---

## Quick Start Example
```julia
using RootSolvers

# Find the root of x^2 - 100^2 using the secant method
sol = find_zero(x -> x^2 - 100^2, SecantMethod(0.0, 1000.0))

if sol.converged
    println("Root found: ", sol.root)
else
    println("Root not found")
end
```

---

## How-to Guide

This guide shows the basic steps for solving a root-finding problem.

### General Workflow (e.g., Secant Method)

#### 1. Define Your Function
Write your function as a Julia callable.
```julia
f(x) = x^3 - 2x - 5
```

#### 2. Choose a Root-Finding Method
Pick a method and provide initial guesses. The type parameter (e.g., `Float64`) is often inferred automatically.
```julia
# For SecantMethod, provide two initial guesses
method = SecantMethod(1.0, 3.0)
```

#### 3. (Optional) Set Tolerance and Solution Type
Customize the convergence criteria and the level of detail in the output.
```julia
# Stop when iterates are closer than 1e-8
tol = SolutionTolerance(1e-8)

# Request detailed output for debugging
soltype = VerboseSolution()
```

#### 4. Call `find_zero`
All arguments after `method` are optional.
```julia
sol = find_zero(f, method, soltype, tol)
```

#### 5. Interpret Results
- `sol.converged`: `true` if a root was found.
- `sol.root`: The root value.
- `sol.err`, `sol.iter_performed`, `sol.root_history` (available with `VerboseSolution`).


### Specific Example: Newton's Method with a Provided Derivative

When using `NewtonsMethod`, you must provide a function that returns both the value `f(x)` and its derivative `f'(x)` as a tuple. This avoids the overhead of automatic differentiation and is highly efficient if you can provide an analytical derivative.

#### 1. Define Function and Derivative
```julia
# This function finds the root of f(x) = x^2 - 4.
# It returns the tuple (f(x), f'(x)).
f_with_deriv(x) = (x^2 - 4, 2x)
```

#### 2. Choose the Method and Call `find_zero`
```julia
# Provide a single initial guess for Newton's method
method = NewtonsMethod(1.0)

# The function f_with_deriv is passed to find_zero
sol = find_zero(f_with_deriv, method)

println("Root found: ", sol.root) # Expected: 2.0
```

---

## High-Performance and GPU Computing 🚀

RootSolvers.jl is designed for high-performance computing, supporting broadcasting over custom data structures and GPU acceleration. This makes it ideal for solving many problems in parallel.

### Broadcasting with Abstract Types
The package works seamlessly with any abstract type that supports broadcasting, making it well-suited for scientific domains like climate modeling.

**Example: Solving over a custom field type**
```julia
using RootSolvers

# Example using regular arrays to represent a field grid
x0 = rand(10, 10)  # A 10x10 field of initial guesses
x1 = x0 .+ 1.0     # A second field of initial guesses

# Define a function that operates element-wise on the field
f(x) = x.^2 .- 2.0

# Solve the root-finding problem across the entire field
method = SecantMethod(x0, x1)
sol = find_zero.(f, method, CompactSolution()) # sol is an Array of structs

# Results
# Use getproperty.() to extract the fields from each struct in the array
converged_field = getproperty.(sol, :converged)
root_field = getproperty.(sol, :root)

println("All converged: ", all(converged_field))
println("Root field shape: ", size(root_field))
```

### GPU Acceleration for Batch Processing
You can achieve significant speedups by running large batches of problems on a GPU.

**GPU Usage Tips:**
- **Use `CompactSolution`:** Only `CompactSolution` is GPU-friendly. `VerboseSolution` is for CPU debugging only.
- **GPU-Compatible Function:** Ensure your function `f(x)` uses only GPU-supported operations.
- **Minimize Data Transfer:** Keep initial guesses and results on the GPU.

**Example: 1 Million problems on the GPU**
```julia
using RootSolvers, CUDA

# Create GPU arrays for batch processing
x0 = CUDA.fill(1.0f0, 1000, 1000)  # 1M initial guesses on GPU
x1 = CUDA.fill(2.0f0, 1000, 1000)  # Second initial guesses

# Define GPU-compatible function
f(x) = x.^3 .- x .- 2.0

# Solve all problems in parallel using broadcasting
method = SecantMethod(x0, x1)
sol = find_zero.(f, method, CompactSolution())

# Results are on the GPU
converged_field = getproperty.(sol, :converged)
root_field = getproperty.(sol, :root)

println("All converged: ", all(converged_field))
println("Root field shape: ", size(root_field))
```

---

## Reference Tables

### Available Root-Finding Methods

| Method | Requirements | Best For |
| :--- | :--- | :--- |
| `SecantMethod` | 2 initial guesses | No derivatives, **fast** convergence|
| `RegulaFalsiMethod` | Bracketing interval | **Guaranteed** convergence |
| `NewtonsMethodAD` | 1 initial guess, differentiable `f` | **Fastest**, uses autodiff, robust step control |
| `NewtonsMethod` | 1 initial guess, `f` and `f'` provided | **Analytical** derivatives, robust step control |

### Available Tolerance Types

| Tolerance Type | Criterion | Best For |
| :--- | :--- | :--- |
| `SolutionTolerance` | `abs(x₂ - x₁)` | When you want iterates to **stabilize** |
| `ResidualTolerance` | `abs(f(x))` | When you want the function value near **zero** |
| `RelativeSolutionTolerance` | `abs((x₂ - x₁)/x₁)` | When root magnitude **varies widely** |
| `RelativeOrAbsolute...`| Relative or Absolute | **Robust** for both small and large roots |

### Available Solution Types

| Solution Type | Features | Best For |
| :--- | :--- | :--- |
| `CompactSolution` | Minimal output, GPU-friendly | **High-performance**, GPU, memory efficiency |
| `VerboseSolution` | Full diagnostics, iteration history | **Debugging**, analysis, CPU |

---

## Troubleshooting
- If not converging, try different initial guesses or a bracketing method (`RegulaFalsiMethod`).
- Use `VerboseSolution()` to inspect the iteration history and diagnose issues.
- Adjust the tolerance for stricter or looser convergence criteria.
