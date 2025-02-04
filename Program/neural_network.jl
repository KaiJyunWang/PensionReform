using Lux, ADTypes, LuxCUDA, Optimisers, Printf, Random, Statistics, Zygote
using Parameters, ComponentArrays, Distributions
using CairoMakie
using Profile, BenchmarkTools
using LinearAlgebra
include("Mortality.jl")
using .Mortality

function set_parameters(; γ = 3f0, β = 0.96f0, R = 1f0/0.96f0, σ_ξ = 0.3f0, δ = [-2f0, 0.13f0, -0.001f0], 
    ρ = 0.97f0, η = 0.9f0, φ_1 = 0.9f0, φ_2 = 20f0, T = life_ceil([1.13, 64200, 0.1, 0.0002]), 
    θ_b = 200f0, κ = 700f0, N = 10000, 
    μ = mortality([1.13, 64200, 0.1, 0.0002]))

    # Monte Carlo draws
    ξ = σ_ξ*randn(N, T) 
    ϵ_n = rand(Gumbel(), N, T, 2) .- Base.MathConstants.γ
    a_0 = 50*rand(N) 
    ϵ_0 = σ_ξ/sqrt(1-ρ^2)*randn(N) 
    n_0 = (rand(N) .> 0.5) 
    p_0 = ones(N)
    wy_0 = zeros(N)
    amis_0 = zeros(N)
    lmis_0 = zeros(N)
    data = permutedims(hcat(a_0, ϵ_0, n_0, p_0, wy_0, amis_0, lmis_0, ones(N)), (2,1)) |> cu

    # Utility 
    u(c,n) = γ == 1 ? η*log(c) + (1-η)*log(1 - φ_1*n) : (c^η*(1 - φ_1*n)^(1-η))^(1-γ)/(1-γ) 
    b(a) = γ == 1 ? θ_b * log(κ + a) : θ_b * (κ + a)^(1-γ)/(1-γ)

    return (γ = γ, β = β, R = R, σ_ξ = σ_ξ, δ = δ, ρ = ρ, η = η, 
        φ_1 = φ_1, φ_2 = φ_2, T = T, θ_b = θ_b, κ = κ, N = N, u = u, b = b, μ = μ, survival = append!([1.0], cumprod(1 .- μ)),
        ξ = cu(ξ), data = data)
end

p = set_parameters()

model = Chain(Dense(8 => 64, sigmoid), Dense(64 => 3, sigmoid))

opt = AMSGrad()

function lifecycle_loss(model, ps, st, states; p = p)
    @unpack γ, β, R, σ_ξ, δ, ρ, η, φ_1, φ_2, T, θ_b, κ, N, u, b, μ, survival, ξ, data = p
    v = 0f0
    for s in 1:T 
        a, ϵ, n, p, wy, amis, lmis, t = states[1, :], states[2, :], states[3, :], states[4, :], states[5, :], states[6, :], states[7, :], states[8, :]
        action, st = Lux.apply(model, dev_gpu(states), ps, st) 
        cp, new_n, add_p = action[1,:], round.(action[2, :]), round.(2f0 * action[3, :])
        disposable = R * a + exp.((δ[1]*cu(ones(N)) + δ[2]*t + δ[3]* t.^2) + ϵ) .* new_n 
        c = cp .* disposable 
        new_a = disposable - c
        new_ϵ = ρ * ϵ + ξ[:, s]
        new_t = t .+ 1
        v += mean(u.(c, new_n) .- φ_2*(new_n - n .== 1) .+ (new_n .* ϵ_n[:,s,1] + (1 .- new_n) .* ϵ_n[:,s,2])) * β^(s-1) * survival[s] + mean(b.(new_a)) * μ[s] * β^s
        states = permutedims(hcat(new_a, new_ϵ, new_n, new_t), (2,1)) |> cu
    end
    return -v, st, ()
end

dev_cpu = cpu_device()
dev_gpu = gpu_device()

ps, st = Lux.setup(Xoshiro(2025), model) |> dev_gpu

lifecycle_loss(model, ps, st, p.data)[1]

tstate = Training.TrainState(model, ps, st, opt)

vjp_rule = AutoZygote()

function main(tstate::Training.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    epoch = 1
    while Zygote.gradient(ps -> lifecycle_loss(model, ps, st, p.data)[1], ps) |> ComponentArray |> norm > 1e-3 && epoch ≤ epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, lifecycle_loss, data, tstate)
        if epoch % 10 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g \t Gradient Norm: %.5g\n" epoch loss Zygote.gradient(ps -> lifecycle_loss(model, ps, st, p.data)[1], ps) |> ComponentArray |> norm
        end
        epoch += 1
    end
    return tstate
end

tstate = main(tstate, vjp_rule, p.data, 700)

Zygote.gradient(ps -> lifecycle_loss(model, ps, st, p.data)[1], ps) |> ComponentArray |> norm
Lux.apply(model, dev_gpu(p.data), ps, st)[1]

# Simulation 
init_a = 0.0
init_n = 1.0
asset = zeros(p.T+1)
asset[1] = init_a
ϵ = zeros(p.T+1)
wage = zeros(p.T)
labor = ones(p.T+1)
for t in 1:p.T 
    action, st = Lux.apply(model, dev_gpu([asset[t], ϵ[t], labor[t], t]), ps, st) 
    cp, labor[t+1] = Array(action)[1], round.(Array(action)[2])
    wage[t] = exp(p.δ[1] + p.δ[2]*t + p.δ[3]*t^2 + ϵ[t])
    disposable = p.R * asset[t] + wage[t] * labor[t+1]
    asset[t+1] = (1-cp) * disposable
    ϵ[t+1] = p.ρ * ϵ[t] + p.σ_ξ * randn()
end

# Plot 
begin 
    fig = Figure(size = (800, 600))
    ax = CairoMakie.Axis(fig[1,1], xlabel = "t")
    #ax2 = CairoMakie.Axis(fig[1,1], yaxisposition = :right) # for intensive margin labor supply 
    #ylims!(ax2, 0.0 ,1.0)
    xlims!(ax, 1, p.T)
    lines!(ax, 1:p.T, asset[2:end], color = :red, linewidth = 2, label = L"a_t")
    lines!(ax, 1:p.T, wage, color = :blue, linewidth = 2, label = L"y_t") 
    lines!(ax, 1:p.T, p.R * asset[1:end-1] + wage .* labor[2:end] - asset[2:end], color = :green, linewidth = 2, label = L"c_t")
    vspan!(ax, findall(isone, labor[2:end]) .- 0.5, findall(isone, labor[2:end]) .+ 0.5, color = (:gray, 0.3))
    #lines!(ax2, 1:p.T, labor[2:end], color = :black, linewidth = 2, label = L"n_t")
    Legend(fig[1,2], ax, framevisible = false)
    fig
end