using Distributions, LinearAlgebra, Plots, Interpolations
using Parameters, Random, Tables, Profile
using CUDA, CUDAKernels, KernelAbstractions, Tullio
using BenchmarkTools

#life-cycle problem of pension solver

#mortality law from Thatcher(1999)
function mortality(x::Array{Float64,1})
    T = floor(Int, ceil((log(1-x[4])+log(x[2])-log(x[1]+x[4]-1))/x[3]))
    μ = ones(T)
    for t in 1:T-1
        μ[t] = x[1]/(1+x[2]*exp(-x[3]*t))+x[4]
    end
    return (T = T, μ = μ)
end

#life Distributions
function life_dist(x::Array{Float64,1}, T::Int)
    D = zeros(T, T) |> x -> LowerTriangular(x)
    for t in 1:T
        for s in t+1:T
            D[s,t] = (1 - sum(D[t:s-1,t]))*x[s]
        end
    end
    return D
end

mort = mortality([1.5, 100000.0, 0.1, 0.0003])

profile = [2.747, 3.48, 4.58]
function aime(rame::Float64, profile::Array{Float64,1} = profile)
    if rame < profile[2]
        return profile[1]
    elseif rame < profile[3]
        return profile[2]
    else
        return profile[3]
    end
end
plot(1.0:0.02:5.0, x -> aime(x), label = "aime")
#pension benefit formula
#unit: month wage
#pension type: 0: no pension, 1: once pension, 2: monthly pension
function benefit(aime::Float64, p::Int64, h::Int64, t::Int64, ra::Int64)
    if (p == 1) || (p == 4)
        return 0.0
    elseif p == 2
        return min(max(h, 2*h-15), 50)*aime
    else
        return max(h*aime*0.00775+3, h*aime*0.0155)*(1+0.04*min(abs(t-ra), 5)*sign(t-ra))
    end
end
plot!(1.0:0.02:5.0, x -> benefit(aime(x), 2, 20, 63, 60), label = "benefit")
#pension tax formula
function tax(aime::Float64, p::Int, τ::Float64 = 0.12)
    if p == 0
        return aime*0.2*τ
    else
        return 0.0
    end
end
plot!(1.0:0.02:5.0, x -> tax(aime(x), 0), label = "tax")
#utility function 
function u(c::Float64, n::Int64; γ::Float64, η::Float64, c_min::Float64)
    if c ≥ c_min
        if γ == 1
            return η*log(c) + (1-η)*log(364-260*n)
        else
            return (((c^η)*(364-260*n)^(1-η))^(1-γ))/(1-γ)
        end
    else
        return -1e200
    end
end

para = @with_kw (γ = 1.5, η = 0.6, r = 0.02, β = 1/(1+r), ϵ = collect(range(0.0, 1.0, 5)), 
    T = 100, μ = mort.μ, init_t = 40, ρ = 0.97, σ = 0.1, ξ = σ*randn(250), a = collect(range(0.0, 25.0, 15)), 
    n = [0,1], h = collect(0:20), h_ceil = 10, c_min = 0.1, δ = [0.1, -0.003, 0.0], φ_l = 1.0, θ_b = 0.5, κ = 2.0,
    aime = profile, p = collect(1:4), ra = 60, τ = 0.12, lme = profile)
para = para()
v = zeros(length(para.a), length(para.ϵ), length(para.h), length(para.n), length(para.aime), length(para.p), length(para.lme), para.T-para.init_t+2)
function solve(v::Array{Float64,8};para)
    (; γ, η, r, β, ϵ, T, μ, init_t, ρ, σ, ξ, a, n, h, h_ceil, c_min, δ, φ_l, θ_b, κ, aime, p, ra, τ, lme) = para
    
    h_comp = [δ[1]*min(h_ceil, h[m]) + δ[2]*min(h_ceil, h[m])^2 for m in 1:length(h)] |> x -> CuArray(x)


    candidate = zeros(length(a), length(ϵ), length(h), length(n), length(aime), length(p), length(lme), length(a), length(n), length(p)) |> x -> CuArray(x)
    current_wage = zeros(length(ϵ), length(h)) |> x -> CuArray(x)
    pension_tax = zeros(length(ϵ), length(h), length(p)) |> x -> CuArray(x)
    pb = zeros(length(h), length(aime), length(p)) |> x -> CuArray(x)
    pension_allow = zeros(length(p), length(p)) |> x -> CuArray(x)
    rpb = zeros(length(h), length(aime), length(p), length(p)) |> x -> CuArray(x)
    consumption = zeros(length(a), length(a), length(n), length(ϵ), length(h), length(n), length(aime), length(p), length(p)) |> x -> CuArray(x)
    flow_utility = zeros(length(a), length(a), length(n), length(ϵ), length(h), length(n), length(aime), length(p), length(p)) |> x -> CuArray(x)
    bequest = zeros(length(a)) |> x -> CuArray(x)
    future_aime = zeros(length(n), length(ϵ), length(h), length(aime), length(lme)) |> x -> CuArray(x)
    future_lme = zeros(length(ϵ), length(h), length(lme)) |> x -> CuArray(x)
    future_utility = zeros(length(a), length(n), length(ϵ), length(h), length(aime), length(p), length(lme)) |> x -> CuArray(x)
    future_value = zeros(length(a), length(n), length(ϵ), length(h), length(aime), length(p), length(lme)) |> x -> CuArray(x)
    pension_tr = zeros(length(p), length(p)) |> x -> CuArray(x)

    policy = zeros(length(a), length(ϵ), length(h), length(n), length(aime), length(p), length(lme), T-init_t+1, 3) |> x -> CuArray(x)



    ϵ_grid = zeros(length(ϵ), T-init_t+2)
    ϵ_grid[:,1] = ϵ
    for t in 2:T-init_t+2
        ϵ_grid[:,t] = range(ρ*ϵ_grid[1,t-1] + minimum(ξ), ρ*ϵ_grid[end,t-1] + maximum(ξ), length(ϵ))
    end
    
    a_cua = CuArray(a)
    n_cua = CuArray(n)
    
    @tullio pension_allow[z,f] = ((p[z] != 1) ? ((((p[z] == 2)||(p[z] == 4)) ? 0 : (((p[z] == 3)&&(p[f] == 3)) ? 1 : 0))) : 1)
    @tullio bequest[j] = $θ_b*(($κ + a_cua[j])^(1 - $γ))/(1 - $γ)
    @tullio pension_tr[z,f] = ((p[f] == 1)||((p[f] == 3)&&(p[z] == 3))||((p[z] == 2)&&(p[z] == 4))||((p[z] == 4)&&(p[z] == 4))) ? 0 : -1e200
    
    for s in 1:T-init_t+1
        t = T-s+1
        mort = μ[t]
        v_func = LinearInterpolation((a, ϵ_grid[:,t-init_t+2], h, n, aime, p, lme), v[:,:,:,:,:,:,:,t-init_t+2])
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])

        #minimum wage
        @tullio current_wage[l,m] = max(2.747, exp(ϵ_cua[l] + h_comp[m]))
        @tullio pension_tax[l,m,f] = (0.2*$τ*(p[f] == 1) + !(p[f] == 1))*current_wage[l,m]
        @tullio pb[m,y,f] = benefit(aime[y], p[f], h[m], $t, $ra)
        @tullio rpb[m,y,z,f] = pb[m,y,f]*pension_allow[z,f]
        @tullio consumption[i,j,k,l,m,x,y,z,f] = (1+$r)*a_cua[i] + current_wage[l,m] * n_cua[k] - a_cua[j] + rpb[m,y,z,f] - $φ_l * (1 - n_cua[x]) * n_cua[k] - pension_tax[l,m,f] 
        @tullio flow_utility[i,j,k,l,m,x,y,z,f] = ((consumption[i,j,k,l,m,x,y,z,f] ≥ $c_min) ? (((consumption[i,j,k,l,m,x,y,z,f])^($η)*(104*n_cua[k] + 364*(1 - n_cua[k]))^(1 - $η))^(1 - $γ))/(1 - $γ) : -1e200)
        @tullio future_aime[k,l,m,y,g] = min(4.58, ((h[m] != 0) ? ((h[m] < 5) ? (h[m]*aime[y] + current_wage[l,m]*n_cua[k])/(h[m] + n_cua[k]) : (aime[y] + 0.2 * max(0, current_wage[l,m]*n_cua[k] - lme[g]))) : aime[y]))
        @tullio future_lme[l,m,g] = min(4.58, max(lme[g], current_wage[l,m]))
        @tullio future_utility[j,k,l,m,y,f,g] = mean(v_func(a_cua[j], $ρ * ϵ_cua[l] .+ ξ, min(h[m] + 1, $h_ceil), n_cua[k], future_aime[k,l,m,y,g], p[f], future_lme[l,m,g]))
        @tullio future_value[j,k,l,m,y,f,g] = $β * (1 - $mort) * future_utility[j,k,l,m,y,f,g] + $β * $mort * bequest[j]

        @tullio candidate[i,l,m,x,y,z,g,j,k,f] = flow_utility[i,j,k,l,m,x,y,z,f] + future_value[j,k,l,m,y,f,g]
        @tullio candidate[i,l,m,x,y,z,g,j,k,f] = candidate[i,l,m,x,y,z,g,j,k,f] + pension_tr[z,f]
 

        can = Array(candidate)
        
        for state in CartesianIndices(v[:,:,:,:,:,:,:,t-init_t+1])
            ind = argmax(can[state,:,:,:])
            policy[state,t-init_t+1,1] = a[ind[1]]
            policy[state,t-init_t+1,2] = n[ind[2]]
            policy[state,t-init_t+1,3] = p[ind[3]]
        end
        
    end
    
    return (policy = policy, ϵ_grid = ϵ_grid)
end

sol = solve(v; para)

plt = plot()
#draw ar(1) log-wage
ϵ = fill(0.2, para.T-para.init_t+1)
h = 0
h_comp = zeros(para.T-para.init_t+1)
a = 5.0*ones(para.T-para.init_t+2)
n = ones(para.T-para.init_t+2)
sim_aime = 2.747*ones(para.T-para.init_t+1)
p = ones(para.T-para.init_t+2)
lme = 2.747*ones(para.T-para.init_t+1)

for t in 2:para.T-para.init_t+2
    a_func = LinearInterpolation((para.a, sol.ϵ_grid[:,t-1], para.h, para.n, para.aime, para.p, para.lme), sol.policy[:,:,:,:,:,:,:,t-1,1])
    a[t] = a_func(a[t-1], ϵ[t-1], h, n[t-1], sim_aime[t-1], p[t-1], lme[t-1])
    n_func = LinearInterpolation((para.a, sol.ϵ_grid[:,t-1], para.h, para.n, para.aime, para.p, para.lme), sol.policy[:,:,:,:,:,:,:,t-1,2])
    n[t] = floor(Int, n_func(a[t-1], ϵ[t-1], h, n[t-1], sim_aime[t-1], p[t-1], lme[t-1]))
    p_func = LinearInterpolation((para.a, sol.ϵ_grid[:,t-1], para.h, para.n, para.aime, para.p, para.lme), sol.policy[:,:,:,:,:,:,:,t-1,3])
    p[t] = floor(Int, p_func(a[t-1], ϵ[t-1], h, n[t-1], sim_aime[t-1], p[t-1], lme[t-1])) #pension type, need more precise
    h = min(h + n[t],para.h[end])
    h_comp[t-1] = para.δ[1]*h + para.δ[2]*h^2 
    ϵ[t-1] = para.ρ*ϵ[t-1] + para.σ*randn()
    sim_aime[t-1] = min(4.58, ((h != 0) ? ((h < 5) ? (h*2.747 + exp(ϵ[t-1] + h_comp[t-1])*n[t-1])/(h + n[t-1]) : (2.747 + 0.2 * max(0, exp(ϵ[t-1] + h_comp[t-1])*n[t-1] - lme[t-1]))) : 2.747))
    lme[t-1] = min(4.58, max(lme[t-1], exp(ϵ[t-1] + h_comp[t-1])))
end
plot!(para.init_t:para.T, exp.(ϵ + h_comp), label = "wage")
plot!(para.init_t:para.T, a[2:end], label = "asset")
vspan!(para.init_t-1 .+ sort(vcat(findall(x -> x == 1, n[2:end]), findall(x -> x == 1, n[2:end]).+1)) .- 0.5, color = :gray, alpha = :0.4, label = "")
consumption = (1+para.r)*a[1:end-1] + (exp.(ϵ + h_comp) .* n[2:end]) - a[2:end] .- para.φ_l*max.(0, (n[2:end] - n[1:end-1]))
findall(x -> x < para.c_min, consumption)
plot!(para.init_t:para.T, consumption, label = "consumption")

