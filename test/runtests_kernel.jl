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
    f′,
    MethodType,
    x_init, x_lower, x_upper,
    tol,
    dst::AbstractArray{FT, N},
) where {FT, N}
    i = @index(Group, Linear)
    @inbounds begin
        # Construct the method and solve the system.
        # Store the solution in `dst`
        method = get_method(MethodType, x_init[i], x_lower[i], x_upper[i], f′)
        sol = find_zero(f, method, CompactSolution(), tol)
        dst[i] = sol.root
    end
end

@testset "CPU/GPU kernel test" begin
    n_elem = problem_size()
    device = get_device()
    work_groups = (1,)
    ndrange = (n_elem,)
    n_failures = 0

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
            for tol in get_tolerances(FT)
                a_dst = Array{FT}(undef, n_elem)
                d_dst = ArrayType(a_dst)
                kernel! = solve_kernel!(device, work_groups)
                event = kernel!(
                    prob.f,
                    prob.f′,
                    MethodType,
                    x_init,
                    x_lower,
                    x_upper,
                    tol,
                    d_dst;
                    ndrange = ndrange
                )
                wait(device, event)

                # if !all(d_dst .≈ prob.x̃)
                #     println("--Problem: $(prob.name), Meth:$MethodType, tol:$tol, FT:$FT")
                #     @show abs.(d_dst .- prob.x̃)
                #     @show d_dst
                #     @show prob.x̃
                #     n_failures += 1
                # else
                #     @test all(d_dst .≈ prob.x̃)
                # end
                @test all(Array(d_dst) .≈ prob.x̃)
            end
        end
    end
    @test n_failures == 0
end
