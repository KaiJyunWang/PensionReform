using Lux, ADTypes, LuxCUDA, Optimisers, Printf, Random, Statistics, Zygote
using Parameters, ComponentArrays, Distributions, FastGaussQuadrature
using CairoMakie
using Profile, BenchmarkTools
using LinearAlgebra, Combinatorics, Tullio, KernelAbstractions

include("Mortality.jl")
using .Mortality

dev_cpu = cpu_device()
dev_gpu = gpu_device()

function data_expansion(;data, A)
    # To work with multidimensional arrays
    if length(size(data)) == 2
        @tullio expansion[i,j,k] := cos(A[i,k]*data[k,j])
    else
        reshaped_data = reshape(data, (size(data, 1), prod(size(data)[2:end])))
        @tullio expansion[i,j,k] := cos(A[i,k]*reshaped_data[k,j])
    end
    return vcat(data, reshape(dropdims(prod(expansion, dims = 3), dims = 3), (size(A, 1), size(data)[2:end]...)))
end

function set_parameters(; β = 0.96f0, R = 1f0/0.96f0, σ = 0.01f0, ρ = 0f0, 
    γ = 1f0, η = 1f0, mort = mortality([1.13, 64200, 0.1, 0.0002]),
    T = 10, N = 10000, σ_ξ = 0f0, ϕ_1 = 0.9f0, ϕ_2 = 0f0, deg = 10)

    ξ, w = gausshermite(10) .|> cu 
    ξ = permutedims(repeat(sqrt(2f0)*σ_ξ*ξ , 1, N), (2,1))
    w = w/sqrt(π) |> cu
    # Smoother for discrete choices
    smoother = σ*(rand(Gumbel(), N, 2) .- MathConstants.γ) |> cu
    
    # Data expansion
    data = cu(rand(4, N))*2f0*π .- π
    data[3, :] = data[3, :] .> 0f0
    data[4, :] .= Float32(π)
    A = hcat([hcat((multiexponents(size(data, 1), i) |> collect)...) for i in 1:deg]...) |> permutedims |> cu .|> Float32
    data = data_expansion(data = data, A = A)

    # Utility function
    u(c, n) = γ == 1 ? η*log(c) + (1-η)*log(1-ϕ_1*n) : (c^η * (1-ϕ_1*n)^(1-η))^(1-γ)/(1-γ) 

    # Transforms
    # [-π, π] -> State space
    a_tran(x) = 500f0*(x + π)/(2f0*π)
    ϵ_tran(x) = quantile(LogNormal(0f0, σ_ξ/sqrt(1-ρ^2)), (x + π)/(2f0*π))
    t_tran(x) = (x + π)*(Float32(T) + 1f0)/(2f0*π)

    # State space -> [-π, π]
    a_tran_inv(y) = 2f0*π*y/500f0 - π
    ϵ_tran_inv(y) = 2f0*π*cdf(LogNormal(0f0, σ_ξ/sqrt(1-ρ^2)), y) - π
    t_tran_inv(y) = 2f0*π*y/(Float32(T) + 1f0) - π

    # Precomputing the law of motions and states
    a = a_tran.(data[1, :])
    ϵ = ϵ_tran.(data[2, :])
    n = data[3, :]
    t = t_tran.(data[4, :]) .- 1f0

    new_ϵ = ρ * repeat(ϵ, 1, 10) + ξ

    return (; β = β, R = R, σ = σ, ρ = ρ, γ = γ, η = η, mort = mort, T = T, N = N, σ_ξ = σ_ξ, 
        ϕ_1 = ϕ_1, ϕ_2 = ϕ_2, ξ = ξ, w = w, data = data, u = u, smoother = smoother, 
        new_ϵ = new_ϵ, dim = size(data, 1), A = A, a = a, ϵ = ϵ, n = n, t = t, 
        a_tran = a_tran, ϵ_tran = ϵ_tran, t_tran = t_tran, 
        a_tran_inv = a_tran_inv, ϵ_tran_inv = ϵ_tran_inv, t_tran_inv = t_tran_inv)
end

p = set_parameters()

function one_period_loss(model_s, ps_s, st_s, data; p = p, model_v = model_v, ps_v = ps_v, st_v = st_v)
    @unpack β, R, σ, ρ, γ, η, mort, T, N, σ_ξ, ϕ_1, ϕ_2, ξ, w, u, smoother, new_ϵ, A, a, ϵ, n, t, a_tran, ϵ_tran, t_tran, a_tran_inv, ϵ_tran_inv, t_tran_inv = p

    cp, labor = model_s(data, ps_s, st_s)[1][1, :], round.(model_s(data, ps_s, st_s)[1][2, :])
    disposable = R*a + exp.(ϵ) .* labor
    new_data = data_expansion(data = permutedims(cat(repeat(disposable .* (1 .- cp), 1, 10), new_ϵ + ξ, repeat(labor, 1, 10), repeat((data[4, :] .|> t_tran_inv) .+ 1f0 .|> t_tran, 1, 10), dims = 3), (3,1,2)), A = A)
    v = u.(disposable .* cp, labor) - min.(labor - n, 0)*ϕ_2 + smoother[:, 1] .* n + smoother[:, 2] .* (1 .- labor) + dropdims(model_v(new_data, ps_v, st_v)[1], dims = 1) * w
    return -mean(v), st_s, ()
end

one_period_loss(model_s, ps_s, st_s, p.data)

# Set up the data for the model_v to approximate.
function get_v_data(model_s, ps_s, st_s, data; p = p, model_v = model_v, ps_v = ps_v, st_v = st_v)
    @unpack β, R, σ, ρ, γ, η, mort, T, N, σ_ξ, ϕ_1, ϕ_2, ξ, w, u, smoother, new_ϵ, A, a, ϵ, n, t, a_tran, ϵ_tran, t_tran, a_tran_inv, ϵ_tran_inv, t_tran_inv = p

    cp, labor = model_s(data, ps_s, st_s)[1][1, :], round.(model_s(data, ps_s, st_s)[1][2, :])
    disposable = R*a + exp.(ϵ) .* labor
    new_data = data_expansion(data = permutedims(cat(repeat(disposable .* (1 .- cp), 1, 10), new_ϵ + ξ, repeat(labor, 1, 10), repeat((data[4, :] .|> t_tran_inv) .+ 1f0 .|> t_tran, 1, 10), dims = 3), (3,1,2)), A = A)
    v = u.(disposable .* cp, labor) - min.(labor - n, 0)*ϕ_2 + smoother[:, 1] .* n + smoother[:, 2] .* (1 .- labor) + dropdims(model_v(new_data, ps_v, st_v)[1], dims = 1) * w
    return (data, reshape(v, 1, N))
end

opt = AMSGrad()

function train_s(tstate::Training.TrainState, vjp, data, epochs; model_v = model_v, ps_v = ps_v, st_v = st_v)
    data = data .|> gpu_device()
    epoch = 1
    gradient_norm = 1f0
    while gradient_norm > 1f-3 && epoch ≤ epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, one_period_loss, data, tstate)
        gradient_norm = Zygote.gradient(ps -> one_period_loss(model_s, ps, st_s, data; p = p, model_v = model_v, ps_v = ps_v, st_v = st_v)[1], ps_s) |> ComponentArray |> norm
        if epoch == epochs || epoch % 50 == 0
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
        epoch += 1
    end
    return tstate
end

const loss_function = MSELoss()

function train_v(tstate::Training.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
        if epoch == epochs || epoch % 50 == 0
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
        epoch += 1
    end
    return tstate
end

# Initialize models 
model_s = Chain(Dense(p.dim => 256, sigmoid), Dense(256 => 256, sigmoid), Dense(256 => 256, sigmoid), Dense(256 => 2, sigmoid))
model_v = Chain(Dense(p.dim => 256, gelu), Dense(256 => 256, gelu), Dense(256 => 256, gelu), Dense(256 => 1))

ps_s, st_s = Lux.setup(Xoshiro(2025), model_s) |> dev_gpu
ps_v, st_v = Lux.setup(Xoshiro(2025), model_v) |> dev_gpu

tstate_v = Training.TrainState(model_v, ps_v, st_v, opt)
tstate_s = Training.TrainState(model_s, ps_s, st_s, opt)
vjp_rule = AutoZygote()

tstate_v = train_v(tstate_v, vjp_rule, (p.data, CUDA.zeros(1, p.N)), 3000)
draws = copy(p.data)
draws[4, :] = 10*CUDA.ones(p.N) .|> p.t_tran_inv
tstate_s = train_s(tstate_s, vjp_rule, draws, 1000; model_v = model_v, ps_v = ps_v, st_v = st_v)
model_s(p.data, ps_s, st_s)[1]
tstate_v = train_v(tstate_v, vjp_rule, get_v_data(model_s, ps_s, st_s, draws), 3000)
model_v(draws, ps_v, st_v)[1]
v_hat = model_v(data_expansion(data = hcat(range(-π + 1f-3, π - 1f-3, length = 100) |> collect |> cu, zeros(100) |> cu, ones(100) |> cu, draws[4, 1:100]) |> permutedims, A = p.A), ps_v, st_v)[1][1, :]
begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1,1])
    lines!(ax, range(-π + 1f-3, π - 1f-3, length = 100) |> collect .|> p.a_tran, Array(v_hat))
    fig
end

# The main solver 
function backward_nn(tstate_s, tstate_v;p = p)
    @unpack β, R, σ, ρ, γ, η, mort, T, N, σ_ξ, ϕ_1, ϕ_2, ξ, w, data, u, smoother, new_ϵ = p

    # Train the value function for the last period 
    tstate_v = train_v(tstate_v, vjp_rule, (p.data, CUDA.zeros(1, N)), 3000)

    draws = copy(data)

    # Main loop 
    for t in T:-1:1
        @printf "Period %d\n" t
        draws[4, :] = t*CUDA.ones(N)
        tstate_s = train_s(tstate_s, vjp_rule, draws, 5000; model_v = model_v, ps_v = ps_v, st_v = st_v)
        tstate_v = train_v(tstate_v, vjp_rule, get_v_data(model_s, ps_s, st_s, draws), 3000)
    end
    return tstate_s, tstate_v
end

tstate_s, tstate_v = backward_nn(tstate_s, tstate_v)

model_s(p.data, ps_s, st_s)[1]

a = collect(1:100) |> cu
ϵ = zeros(100) |> cu
n = ones(100) |> cu
t = cu(ones(100))
data = stack((a, ϵ, n, t), dims = 1)
model_s(data, ps_s, st_s)[1]
begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1,1])
    lines!(ax, Array(a), Array(vec(model_v(data, ps_v, st_v)[1])))
    fig
end
