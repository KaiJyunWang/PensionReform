using Lux, ADTypes, LuxCUDA, Optimisers, Printf, Random, Statistics, Zygote
using Parameters
using CairoMakie

function set_parameters(; γ = 3f0, β = 0.96f0, R = 1f0/0.96f0, σ_ξ = 0.3f0, 
    ρ = 0.97f0, η = 0.9f0, φ_1 = 0.5f0, φ_2 = 3.0f0, T = 100, θ_b = 0.0f0, κ = 700.0f0, N = 1000)

    # Monte Carlo draws
    ξ = σ_ξ*randn(N, T) 
    a_0 = 10*rand(N) 
    ϵ_0 = σ_ξ/sqrt(1-ρ^2)*randn(N) 
    n_0 = (rand(N) .> 0.5) 
    data = transpose(hcat(a_0, ϵ_0, n_0, ones(N))) |> Matrix |> cu

    # Utility 
    u(c,n) = γ == 1 ? η*log(c) + (1-η)*log(1 - φ_1*n) : (c^η*(1 - φ_1*n)^(1-η))^(1-γ)/(1-γ) 

    return (γ = γ, β = β, R = R, σ_ξ = σ_ξ, ρ = ρ, η = η, 
        φ_1 = φ_1, φ_2 = φ_2, T = T, θ_b = θ_b, κ = κ, N = N, u = u, 
        ξ = cu(ξ), a_0 = a_0, ϵ_0 = ϵ_0, n_0 = n_0, data = data)
end

p = set_parameters()

model = Chain(Dense(4 => 16, sigmoid), Dense(16 => 2, sigmoid))

opt = Adam(0.03f0)

function lifecycle_loss(model, ps, st, states; p = p)
    @unpack γ, β, R, σ_ξ, ρ, η, φ_1, φ_2, T, θ_b, κ, N, u, ξ, a_0, ϵ_0, n_0, data = p
    
    v = 0.0
    for s in 1:T 
        a, ϵ, n, t = states[1, :], states[2, :], states[3, :], states[4, :]
        action, st = Lux.apply(model, dev_gpu(states), ps, st) 
        ap, new_n = action[1,:], round.(action[2, :])
        disposable = R * a + exp.(ϵ) .* new_n
        new_a = ap .* disposable
        new_ϵ = ρ * ϵ + ξ[:, s]
        new_t = t .+ 1
        v += mean(u.(disposable - new_a, new_n)) * β^(s-1)
        states = permutedims(hcat(new_a,new_ϵ,new_n,new_t), (2,1)) |> cu
    end
    return -v, st, ()
end

dev_cpu = cpu_device()
dev_gpu = gpu_device()

ps, st = Lux.setup(rng, model) |> dev_gpu

lifecycle_loss(model, ps, st, p.data)

tstate = Training.TrainState(model, ps, st, opt)

vjp_rule = AutoZygote()

Training.single_train_step!(vjp_rule, lifecycle_loss, p.data, tstate)

function main(tstate::Training.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, lifecycle_loss, data, tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return tstate
end

tstate = main(tstate, vjp_rule, p.data, 1000)
Lux.apply(model, dev_gpu(p.data), ps, st)[1]

# Simulation 
init_a = 5.0
init_n = 1.0
asset = zeros(p.T+1)
asset[1] = init_a
ϵ = zeros(p.T+1)
labor = ones(p.T+1)
for t in 1:p.T 
    action, st = Lux.apply(model, dev_gpu([asset[t], ϵ[t], labor[t], t]), ps, st) 
    ap, labor[t+1] = Array(action)[1], round(Array(action)[2]) 
    disposable = p.R * asset[t] + exp(ϵ[t]) * labor[t+1]
    asset[t+1] = ap * disposable
    ϵ[t+1] = p.ρ * ϵ[t] + p.σ_ξ * randn()
end

# Plot 
begin 
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1,1], xlabel = "t")
    lines!(ax, 1:p.T, asset[2:end], color = :red, linewidth = 2, label = L"a_t")
    lines!(ax, 1:p.T, exp.(ϵ[2:end]), color = :blue, linewidth = 2, label = L"y_t") 
    lines!(ax, 1:p.T, p.R * asset[1:end-1] + exp.(ϵ[1:end-1]) .* labor[2:end] - asset[2:end], color = :green, linewidth = 2, label = L"c_t")
    Legend(fig[1,2],ax,  framevisible = false)
    fig
end