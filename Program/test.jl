using Lux, Statistics, Zygote, Random, Printf, Distributions
using CairoMakie, Optimisers, ADTypes, LaTeXStrings, Parameters

# Packages for backward induction
using Optim, Interpolations

# Set the seed 
seed = 123
rng = Xoshiro(seed)

# Parameters 
function set_parameters(; β = 0.95, γ = 1.5, σ = 0.3, ρ = 0.0, T = 20, r = 1/ 0.95 - 1)
    ξ = randn(1000, T)
    return (β = β, γ = γ, σ = σ, ρ = ρ, T = T, r = r)
end

p = set_parameters()

# Utility Flow 
# Numerical stability
u(c; γ = p.γ, β = p.β) = c ≥ 0 ? ((γ == 1 ? log(c) : c^(1-γ)/(1-γ))*(1-β)) : -1e10

# Sampler 
function sample(n; a_0_dist = Uniform(-5, 5), ϵ_0_dist = Uniform(-1, 1), T = p.T, rng = rng, σ = p.σ)
    a_0 = rand(rng, a_0_dist, 1, n) .|> Float32
    ϵ_0 = rand(rng, ϵ_0_dist, 1, n) .|> Float32
    t = zeros(1, n) .|> Float32
    ξ = σ*randn(Float32, 1, n) 
    return vcat(a_0, ϵ_0, t), ξ
end

data, ξ = sample(100)

# Neural network 
model = Chain(Dense(3 => 32, selu), Dense(32 => 32, selu), Dense(32 => 1, softplus))

ps, st = Lux.setup(rng, model)

# Optimizer
opt = AMSGrad(0.0001)

# Auto-differentiation
vjp_rule = AutoZygote()

# Customized penalty for transversality condition
penalty(x) = 100*mean(abs2.(x))

# Customized loss function 
# Using -v as the loss function

function loss_function(model, ps, st, data; p = p)
    @unpack β, γ, σ, ρ, T, r = p
    v = 0f0
    n = size(data, 2)
    # Compute the value
    for s in 1:T
        c, st = Lux.apply(model, data, ps, st) 
        a, ϵ, t = data[1:1, :], data[2:2, :], data[3:3, :]
        new_ϵ = ρ*ϵ + σ*randn(Float32, 1, n)
        new_a = (1+r)*a + exp.(ϵ) - c
        new_t = t .+ 1
        v += mean(u.(c))*(β^(s-1))
        data = vcat(new_a, new_ϵ, new_t)
    end
    # constraint for borrowing
    return - v + penalty(data[1, :]), st, ()
end

# Training state 
tstate = Training.TrainState(model, ps, st, opt)

# Training function 
function train(tstate, vjp, data, loss_function, epochs)
    epoch = 1
    #loss = Inf
    #distance = Inf
    while epoch ≤ epochs #&& distance > 1f-5
        _, new_loss, _, tstate = Training.single_train_step!(
            vjp, loss_function, data, tstate)
        if epoch % 100 == 1 || epoch == epochs
            @printf "Epoch: %3d \t New Loss: %.5g\n" epoch new_loss
        end
        #=
        if epoch % 100 == 1 
            data, ξ = sample(100)
        end
        =#
        #distance = abs(loss - new_loss)
        #loss = new_loss
        epoch += 1
    end
    return tstate
end

# Pretraining the model
#tstate = train(tstate, vjp_rule, data, pretrain_loss, 500)

# Train the model 
tstate = train(tstate, vjp_rule, data, loss_function, 20000)
#=
# A comparison to the traditional backward induction method 
function backward(; β = β, r = r, γ = γ, σ = σ, ρ = ρ, T = T, tol = 1e-10)
    a_grid, ϵ_grid = collect(-5:0.1:5), collect(-1:0.1:1)
    n_a, n_ϵ = length(a_grid), length(ϵ_grid)
    n_ξ = 100
    ξ = σ*randn(n_ξ)
    v = zeros(n_a, n_ϵ, T+1)
    c = zeros(n_a, n_ϵ, T) 
    for s in reverse(1:T)
        v_func = LinearInterpolation((a_grid, ϵ_grid), v[:, :, s+1], extrapolation_bc = Line()) 
        for (i, a) in enumerate(a_grid), (j, ϵ) in enumerate(ϵ_grid)
            obj(c) = u(c[1]) + β * mean(v_func.((a*(1+r) + exp(ϵ) - c[1])*ones(n_ξ), ρ*ϵ .+ ξ)) + 1e5*(s == T)*min((a*(1+r) + exp(ϵ) - c[1]), 0)
            res = maximize(obj, [exp(ϵ)])
            v[i, j, s], c[i, j, s] = Optim.maximum(res), Optim.maximizer(res)[1]
        end
    end
    return (; v = v, c = c)
end

back_v, back_c = backward()
=#
# Simulate Results
a = zeros(1, p.T+1) .|> Float32
ϵ = zeros(1, p.T+1) .|> Float32 
time = transpose(repeat(collect(1:p.T), 1, 1)) .|> Float32
c = zeros(1, p.T) .|> Float32
vfi_c = zeros(T)
vfi_a = zeros(T+1)
for s in 1:p.T 
    #back_c_func = LinearInterpolation((collect(-5:0.1:5), collect(-1:0.1:1)), back_c[:, :, s], extrapolation_bc = Line())
    #vfi_c[s] = back_c_func(vfi_a[s], ϵ[1, s])
    c[:, s:s] = Lux.apply(tstate.model, vcat(a[:, s:s], ϵ[:, s:s], time[:, s:s]), tstate.parameters, tstate.states)[1]
    a[:, s+1:s+1] = (1+r)*a[:, s:s] + exp.(ϵ[:, s:s]) - c[:, s:s]
    #vfi_a[s+1] = (1+r)*vfi_a[s] + exp.(ϵ[1, s]) - vfi_c[s]
    ϵ[:, s+1:s+1] = ρ*ϵ[:, s:s] + p.σ*randn(Float32, 1, 1)
end

# Plot the results
begin 
    step = tstate.step
    fig2 = Figure()
    ax2 = Axis(fig2[1,1], xlabel = L"t", title = "Life-Cycle Profile") 
    lines!(ax2, vec(time), vec(c), color = :green, label = L"c_t\quad (RL)")
    #lines!(ax2, vec(time), vfi_c, color = :brown, label = L"c_t\quad (VFI)")
    lines!(ax2, vec(time), exp.(ϵ[1,1:end-1]), color = :blue, label = L"y_t")
    lines!(ax2, vec(time), a[1,2:end], color = :red, label = L"a_t\quad (RL)")
    #lines!(ax2, vec(time), vfi_a[2:end], color = :purple, label = L"a_t\quad (VFI)")
    Legend(fig2[1, 2], ax2, "Steps: $step", framevisible = false)
    fig2
end