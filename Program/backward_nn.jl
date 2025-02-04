using Lux, ADTypes, LuxCUDA, Optimisers, Printf, Random, Statistics, Zygote
using Parameters, ComponentArrays, Distributions, FastGaussQuadrature
using CairoMakie
using Profile, BenchmarkTools
using LinearAlgebra

include("Mortality.jl")
using .Mortality

dev_cpu = cpu_device()
dev_gpu = gpu_device()

model_s = Chain(Dense(4 => 64, sigmoid), Dense(64 => 2, sigmoid))
model_v = Chain(Dense(4 => 64, relu), Dense(64 => 64, relu), Dense(64 => 1))

ps_s, st_s = Lux.setup(Xoshiro(2025), model_s) |> dev_gpu
ps_v, st_v = Lux.setup(Xoshiro(2025), model_v) |> dev_gpu

function set_parameters(; β = 0.96f0, R = 1f0/0.96f0, σ = 0.01f0, ρ = 0.97f0, 
    γ = 3f0, η = 0.9f0, mort = mortality([1.13, 64200, 0.1, 0.0002]),
    T = life_ceil([1.13, 64200, 0.1, 0.0002]), N = 10000, σ_ξ = 0.3f0, 
    ϕ_1 = 0.762f0, ϕ_2 = 5f0)

    ξ, w = gausshermite(10) .|> cu 
    ξ = permutedims(repeat(sqrt(2f0)*σ_ξ*ξ , 1, N), (2,1))
    w = w/sqrt(π) |> cu
    # Monte Carlo 
    a = 1000*rand(N) |> cu
    ϵ = σ_ξ/sqrt(1-ρ^2)*randn(N) |> cu
    n = (rand(N) .> 0.5) |> cu
    smoother = σ*(rand(Gumbel(), N, 2) .- MathConstants.γ) |> cu

    data = stack((a, ϵ, n, T*cu(ones(N))), dims = 1) 

    u(c, n) = γ == 1 ? η*log(c) + (1-η)*log(1-ϕ_1*n) : (c^η * (1-ϕ_1*n)^(1-η))^(1-γ)/(1-γ) 

    # Precomputing the law of motions
    new_ϵ = ρ * repeat(ϵ, 1, 10) + ξ

    return (; β = β, R = R, σ = σ, ρ = ρ, γ = γ, η = η, mort = mort, T = T, N = N, σ_ξ = σ_ξ, 
        ϕ_1 = ϕ_1, ϕ_2 = ϕ_2, ξ = ξ, w = w, data = data, u = u, smoother = smoother, 
        new_ϵ = new_ϵ)
end

p = set_parameters()

function one_period_loss(model_s, ps_s, st_s, data; p = p, model_v = model_v, ps_v = ps_v, st_v = st_v)
    @unpack β, R, σ, ρ, γ, η, mort, T, N, σ_ξ, ϕ_1, ϕ_2, ξ, w, u, smoother, new_ϵ = p

    a, ϵ, n, t = data[1, :], data[2, :], data[3, :], data[4, :]
    cp, labor = model_s(data, ps_s, st_s)[1][1, :], round.(model_s(data, ps_s, st_s)[1][2, :])
    disposable = R*a + exp.(ϵ) .* labor
    new_data = permutedims(cat(repeat(disposable .* (1 .- cp), 1, 10), new_ϵ + ξ, repeat(labor, 1, 10), repeat(t .+ 1, 1, 10), dims = 3), (3,1,2)) |> cu
    v = u.(disposable .* cp, labor) - min.(labor - n, 0)*ϕ_2 + smoother[:, 1] .* n + smoother[:, 2] .* (1 .- labor) + dropdims(model_v(new_data, ps_v, st_v)[1], dims = 1) * w
    return -mean(v), st_s, ()
end

# Set up the data for the model_v to approximate.
function get_v_data(model_s, ps_s, st_s, data; p = p, model_v = model_v, ps_v = ps_v, st_v = st_v)
    @unpack β, R, σ, ρ, γ, η, mort, T, N, σ_ξ, ϕ_1, ϕ_2, ξ, w, u, smoother, new_ϵ = p

    a, ϵ, n, t = data[1, :], data[2, :], data[3, :], data[4, :]
    cp, labor = model_s(data, ps_s, st_s)[1][1, :], round.(model_s(data, ps_s, st_s)[1][2, :])
    disposable = R*a + exp.(ϵ) .* labor
    new_data = permutedims(cat(repeat(disposable .* (1 .- cp), 1, 10), new_ϵ + ξ, repeat(labor, 1, 10), repeat(t .+ 1, 1, 10), dims = 3), (3,1,2)) |> cu
    v = u.(disposable .* cp, labor) - min.(labor - n, 0)*ϕ_2 + smoother[:, 1] .* n + smoother[:, 2] .* (1 .- labor) + β * dropdims(model_v(new_data, ps_v, st_v)[1], dims = 1) * w
    return (data, reshape(v, 1, N))
end

opt = AMSGrad()

function train_s(tstate::Training.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    epoch = 1
    gradient_norm = 1f0
    while gradient_norm > 1f-3 && epoch ≤ epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, one_period_loss, data, tstate)
        gradient_norm = Zygote.gradient(ps -> one_period_loss(model_s, ps, st_s, data; p = p)[1], ps_s) |> ComponentArray |> norm
        if epoch % 10 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g \t Gradient Norm: %.5g\n" epoch loss gradient_norm
        end
        epoch += 1
    end
    return tstate_s
end

const loss_function = MSELoss()

# Loss function for the value function approximation
function v_loss(model_v, ps_v, st_v, data)
    state, v = data
    v_pred = model_v(state, ps_v, st_v)[1]
    return loss_function(v, v_pred), st_v, ()
end

function train_v(tstate::Training.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, v_loss, data, tstate)
        if epoch % 10 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss 
        end
        epoch += 1
    end
    return tstate
end

tstate_v = Training.TrainState(model_v, ps_v, st_v, opt)
tstate_s = Training.TrainState(model_s, ps_s, st_s, opt)
vjp_rule = AutoZygote()

# The main solver 
function backward_nn()
    
end


tstate_s = train_s(tstate_s, vjp_rule, p.data, 1000)

tstate_v = train_v(tstate_v, vjp_rule, get_v_data(model_s, ps_s, st_s, p.data), 500)
