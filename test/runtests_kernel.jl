if get(ARGS, 1, "Array") == "CuArray"
    using CUDA
    using CUDAKernels
    ArrayType = CUDA.CuArray
    CUDA.allowscalar(false)
    get_device() = CUDADevice()
else
    ArrayType = Array
    get_device() = CPU()
end

@show ArrayType

# Re-include test_helper so we can run
# this file on its own.
include("test_helper.jl")

using KernelAbstractions

# only grab array problems
filter!(x->x.x_init isa AbstractArray, problem_list)

@kernel function solve_kernel!(
    f,
    ff′,
    MethodType,
    x_init, x_lower, x_upper,
    cc,
    dst::AbstractArray{FT, N},
) where {FT, N}
    i = @index(Group, Linear)
    @inbounds begin
        # Construct the method and solve the system.
        # Store the solution in `dst`
        method = get_method(MethodType, x_init[i], x_lower[i], x_upper[i])
        _f = MethodType isa NewtonsMethodType ? ff′ : f
        sol = find_zero(_f, method, CompactSolution(), cc)
        dst[i] = sol.root
    end
end

@testset "CPU/GPU kernel test" begin
    n_elem = problem_size()
    device = get_device()
    work_groups = (1,)
    ndrange = (n_elem,)

    for prob in problem_list
        FT = typeof(prob.x̃)
        x_init = prob.x_init
        x_lower = prob.x_lower
        x_upper = prob.x_upper
        for MethodType in (
                    SecantMethodType(),
                    RegulaFalsiMethodType(),
                    NewtonsMethodADType(),
                    NewtonsMethodType(),
                )
            for cc in get_convergence_criteria(RootSolvers.maxiters_default, FT)
                a_dst = Array{FT}(undef, n_elem)
                d_dst = ArrayType(a_dst)
                kernel! = solve_kernel!(device, work_groups)
                event = kernel!(
                    prob.f,
                    prob.ff′,
                    MethodType,
                    x_init,
                    x_lower,
                    x_upper,
                    cc,
                    d_dst;
                    ndrange = ndrange
                )
                wait(device, event)

                @test all(Array(d_dst) .≈ prob.x̃)
            end
        end
    end
end
