var documenterSearchIndex = {"docs":
[{"location":"Installation/#Installation","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"Installation/","page":"Installation","title":"Installation","text":"RootSolvers.jl is a Julia registered package, and can be added from the Julia Pkg manager:","category":"page"},{"location":"Installation/","page":"Installation","title":"Installation","text":"(v1.x) pkg> add RootSolvers","category":"page"},{"location":"#RootSolvers.jl","page":"Home","title":"RootSolvers.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A simple GPU-capable root solver package.","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using RootSolvers\n\nsol = find_zero(x -> x^2 - 100^2,\n                SecantMethod{Float64}(0.0, 1000.0),\n                CompactSolution());","category":"page"},{"location":"API/#API","page":"API","title":"API","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"CurrentModule = RootSolvers","category":"page"},{"location":"API/#RootSolvers","page":"API","title":"RootSolvers","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"RootSolvers","category":"page"},{"location":"API/#RootSolvers.RootSolvers","page":"API","title":"RootSolvers.RootSolvers","text":"RootSolvers\n\nContains functions for solving roots of non-linear equations. See find_zero.\n\nExample\n\njulia> using RootSolvers\n\njulia> sol = find_zero(x -> x^2 - 100^2,\n                       SecantMethod{Float64}(0.0, 1000.0),\n                       CompactSolution());\n\njulia> sol\nRootSolvers.CompactSolutionResults{Float64}(99.99999999994358, true)\n\n\n\n\n\n","category":"module"},{"location":"API/#Numerical-methods","page":"API","title":"Numerical methods","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"find_zero\nNewtonsMethodAD\nNewtonsMethod\nRegulaFalsiMethod\nSecantMethod","category":"page"},{"location":"API/#RootSolvers.find_zero","page":"API","title":"RootSolvers.find_zero","text":"sol = find_zero(\n        f::F,\n        method::RootSolvingMethod{FT},\n        soltype::SolutionType,\n        tol::Union{Nothing, AbstractTolerance} = nothing,\n        maxiters::Int = 10_000,\n        )\n\nFinds the nearest root of f. Returns a the value of the root x such that f(x) ≈ 0, and a Boolean value converged indicating convergence.\n\nf function of the equation to find the root\nmethod can be one of:\nSecantMethod(): Secant method\nRegulaFalsiMethod(): Regula Falsi method\nNewtonsMethodAD(): Newton's method using Automatic Differentiation\nNewtonsMethod(): Newton's method\nsoltype is a solution type which may be one of:    CompactSolution GPU-capable. Solution has converged and root only, see CompactSolutionResults    VerboseSolution CPU-only. Solution has additional diagnostics, see VerboseSolutionResults\ntol is a tolerance type to determine when to stop iterations.\nmaxiters is the maximum number of iterations.\n\n\n\n\n\n","category":"function"},{"location":"API/#RootSolvers.NewtonsMethodAD","page":"API","title":"RootSolvers.NewtonsMethodAD","text":"NewtonsMethodAD\n\nFields\n\nx0\ninitial guess\n\n\n\n\n\n","category":"type"},{"location":"API/#RootSolvers.NewtonsMethod","page":"API","title":"RootSolvers.NewtonsMethod","text":"NewtonsMethod\n\nFields\n\nx0\ninitial guess\nf′\nf′ derivative of function f whose zero is sought\n\n\n\n\n\n","category":"type"},{"location":"API/#RootSolvers.RegulaFalsiMethod","page":"API","title":"RootSolvers.RegulaFalsiMethod","text":"RegulaFalsiMethod\n\nFields\n\nx0\nlower bound\nx1\nupper bound\n\n\n\n\n\n","category":"type"},{"location":"API/#RootSolvers.SecantMethod","page":"API","title":"RootSolvers.SecantMethod","text":"SecantMethod\n\nFields\n\nx0\nlower\nx1\nupper bound\n\n\n\n\n\n","category":"type"},{"location":"API/#Solution-types","page":"API","title":"Solution types","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"CompactSolution\nVerboseSolution\nVerboseSolutionResults\nCompactSolutionResults","category":"page"},{"location":"API/#RootSolvers.CompactSolution","page":"API","title":"RootSolvers.CompactSolution","text":"CompactSolution <: SolutionType\n\nUsed to return a CompactSolutionResults\n\n\n\n\n\n","category":"type"},{"location":"API/#RootSolvers.VerboseSolution","page":"API","title":"RootSolvers.VerboseSolution","text":"VerboseSolution <: SolutionType\n\nUsed to return a VerboseSolutionResults\n\n\n\n\n\n","category":"type"},{"location":"API/#RootSolvers.VerboseSolutionResults","page":"API","title":"RootSolvers.VerboseSolutionResults","text":"VerboseSolutionResults{FT} <: AbstractSolutionResults{FT}\n\nResult returned from find_zero when VerboseSolution is passed as the soltype.\n\n\n\n\n\n","category":"type"},{"location":"API/#RootSolvers.CompactSolutionResults","page":"API","title":"RootSolvers.CompactSolutionResults","text":"CompactSolutionResults{FT} <: AbstractSolutionResults{FT}\n\nResult returned from find_zero when CompactSolution is passed as the soltype.\n\n\n\n\n\n","category":"type"},{"location":"API/#Tolerance-types","page":"API","title":"Tolerance types","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"ResidualTolerance\nSolutionTolerance\nRelativeSolutionTolerance","category":"page"},{"location":"API/#RootSolvers.ResidualTolerance","page":"API","title":"RootSolvers.ResidualTolerance","text":"ResidualTolerance\n\nA tolerance type based on the residual of the equation f(x) = 0\n\n\n\n\n\n","category":"type"},{"location":"API/#RootSolvers.SolutionTolerance","page":"API","title":"RootSolvers.SolutionTolerance","text":"SolutionTolerance\n\nA tolerance type based on the solution x of the equation f(x) = 0\n\n\n\n\n\n","category":"type"},{"location":"API/#RootSolvers.RelativeSolutionTolerance","page":"API","title":"RootSolvers.RelativeSolutionTolerance","text":"RelativeSolutionTolerance\n\nA tolerance type based on consecutive iterations of solution x of the equation f(x) = 0\n\n\n\n\n\n","category":"type"},{"location":"API/#Internal-helper-methods","page":"API","title":"Internal helper methods","text":"","category":"section"},{"location":"API/","page":"API","title":"API","text":"value_deriv\nmethod_args","category":"page"},{"location":"API/#RootSolvers.value_deriv","page":"API","title":"RootSolvers.value_deriv","text":"value_deriv(f, x)\n\nCompute the value and derivative f(x) using ForwardDiff.jl.\n\n\n\n\n\n","category":"function"},{"location":"API/#RootSolvers.method_args","page":"API","title":"RootSolvers.method_args","text":"method_args(method::RootSolvingMethod)\n\nReturn tuple of positional args for RootSolvingMethod.\n\n\n\n\n\n","category":"function"}]
}