push!(LOAD_PATH, joinpath(@__DIR__, "..", "env", "Plots")) # add Plots env

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate(;verbose=true)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))

using RootSolvers, Documenter

pages = Any[
    "Home" => "index.md",
    "Installation" => "Installation.md",
    "API" => "API.md",
]

mathengine = MathJax(Dict(
    :TeX => Dict(
        :equationNumbers => Dict(:autoNumber => "AMS"),
        :Macros => Dict(),
    ),
))

format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    mathengine = mathengine,
    collapselevel = 1,
)

makedocs(
    sitename = "RootSolvers.jl",
    format = format,
    strict = true,
    clean = true,
    modules = [RootSolvers],
    pages = pages,
)

deploydocs(
    repo = "github.com/CliMA/RootSolvers.jl.git",
    target = "build",
    push_preview = true,
)
