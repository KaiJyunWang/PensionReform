using Distributions, LinearAlgebra, Plots, Interpolations
using Parameters, Random, Tables, Profile
using CUDA, CUDAKernels, KernelAbstractions, Tullio

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
plot(1:mort.T, mort.μ, label = "mortality")
dist = life_dist(mort.μ, mort.T)
findall(x -> x ≈ 1, sum(dist, dims = 1))
plt = plot()
for t in 1:mort.T-30
    plot!(plt, t+1:mort.T, dist[t:end,t], label = "", color = RGBA(t/mort.T, 0, 1-t/mort.T, 0.5))
end
plot(41:mort.T, dist[41:end,40], label = "dist")
#pension benefit formula
function benefit(ame::Array{Float64,1}, p::Int)
    
end

#pension tax formula
function tax(ame::Array{Float64,1}, p::Int)
    
end

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

para = @with_kw (γ = 1.5, η = 0.6, r = 0.02, β = 1/(1+r), ϵ = collect(range(0.0, 1.0, 11)), 
    T = mort.T, μ = mort.μ, init_t = 31, ρ = 0.97, σ = 0.1, ξ = σ*randn(250), a = collect(range(0.0, 15.0, 21)), 
    n = [0,1], h = collect(0:15), c_min = 1e-5, δ = [0.1, -0.003, -0.002], φ_l = 1.0)
para = para()
v = zeros(length(para.a), length(para.ϵ), length(para.h), length(para.n), para.T-para.init_t+1)
function solve(v::Array{Float64,5};para)
    (; γ, η, r, β, ϵ, T, μ, init_t, ρ, σ, ξ, a, n, h, c_min, δ, φ_l) = para

    h_ceil = h[end]
    h_comp = [δ[1]*h[m] + δ[2]*h[m]^2 for m in 1:length(h)] |> x -> CuArray(x)


    candidate = zeros(length(a), length(a), length(n), length(ϵ), length(h), length(n)) |> x -> CuArray(x)
    policy = zeros(length(a), length(ϵ), length(h), length(n), T-init_t+1, 2)
    v1 = zeros(length(ϵ), length(h)) |> x -> CuArray(x)
    v2 = zeros(length(ϵ), length(h), length(n)) |> x -> CuArray(x)
    v3 = zeros(length(a), length(a), length(n), length(ϵ), length(h), length(n)) |> x -> CuArray(x)    
    v4 = zeros(length(a), length(a), length(n), length(ϵ), length(h), length(n)) |> x -> CuArray(x)
    v5 = zeros(length(a), length(n), length(ϵ), length(h)) |> x -> CuArray(x)
    v6 = zeros(length(h), length(n)) |> x -> CuArray(x)
    v7 = zeros(length(n), length(n)) |> x -> CuArray(x)

    ϵ_grid = zeros(length(ϵ), T-init_t+1)
    ϵ_grid[:,1] = ϵ
    for t in 2:T-init_t+1
        ϵ_grid[:,t] = range(ρ*ϵ_grid[1,t-1] + minimum(ξ), ρ*ϵ_grid[end,t-1] + maximum(ξ), length(ϵ))
    end
    
    a_cua = CuArray(a)
    n_cua = CuArray(n)
    @tullio v6[m,k] = min(h[m]+n[k], $h_ceil)
    @tullio v7[k,x] = max(0, n[k]-n[x])
    
    for s in 1:T-init_t
        t = T-s
        v_func = LinearInterpolation((a, ϵ_grid[:,t-init_t+2], h, n), v[:,:,:,:,t-init_t+2])
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        age_prod = δ[3]*t
        @tullio v1[l,m] = exp(ϵ_cua[l] + h_comp[m] + $age_prod)
        @tullio v2[l,m,k] = v1[l,m]*n[k]
        @tullio v3[i,j,k,l,m,x] = (1+$r)*a_cua[i] + v2[l,m,k] - a_cua[j] - $φ_l*v7[k,x]
        @tullio v4[i,j,k,l,m,x] = u(v3[i,j,k,l,m,x], n[k]; γ = γ, η = η, c_min = c_min)
        @tullio v5[j,k,l,m] = $β*(1-μ[$t])*mean(v_func(a_cua[j], $ρ*ϵ_cua[l] .+ ξ, v6[m,k], n[k]))
        @tullio candidate[i,j,k,l,m,x] = v4[i,j,k,l,m,x] + v5[j,k,l,m]
        can = Array(candidate)
        
        for i in 1:length(a), l in 1:length(ϵ), m in 1:length(h), x in 1:length(n)
            v[i,l,m,x,t-init_t+1], ind = findmax(can[i,:,:,l,m,x])
            policy[i,l,m,x,t-init_t+1,1] = a[ind[1]]
            policy[i,l,m,x,t-init_t+1,2] = n[ind[2]]
        end
        
    end
    
    return (policy = policy, v = v, ϵ_grid = ϵ_grid)
end

sol = solve(v; para)

plt = plot()
#draw ar(1) log-wage
ϵ = fill(0.1, para.T+1)
h = 0
h_comp = zeros(para.T)
a = 0.1*ones(para.T+1)
n = zeros(para.T)

for t in 2:para.T-para.init_t
    ϵ[t] = para.ρ*ϵ[t-1] + para.σ*randn()
    a_func = LinearInterpolation((para.a, sol.ϵ_grid[:,t], para.h, para.n), sol.policy[:,:,:,:,t-1,1])
    a[t] = a_func(a[t-1], ϵ[t-1], h, n[t-1])
    n_func = LinearInterpolation((para.a, sol.ϵ_grid[:,t], para.h, para.n), sol.policy[:,:,:,:,t-1,2])
    n[t] = floor(Int, n_func(a[t-1], ϵ[t-1], h, n[t-1]))
    h = min(h + n[t],para.h[end])
    h_comp[t] = para.δ[1]*h + para.δ[2]*h^2 + para.δ[3]*(para.init_t+t-2)
end
plot!(plt, para.init_t:para.T-1, exp.(ϵ[2:para.T-para.init_t+1] + h_comp[1:para.T-para.init_t]), label = "wage")
plot!(plt, para.init_t:para.T-1, a[2:para.T-para.init_t+1], label = "asset")
vspan!(para.init_t-1 .+ sort(vcat(findall(x -> x == 1, n[2:para.T-para.init_t+1]), findall(x -> x == 1, n[2:para.T-para.init_t+1]).+1)) .- 0.5, color = :gray, alpha = :0.4, label = "")
consumption = (1+para.r)*a[1:para.T-para.init_t] + (exp.(ϵ[2:para.T-para.init_t+1] + h_comp[1:para.T-para.init_t]) .* n[2:para.T-para.init_t+1]) - a[2:para.T-para.init_t+1] .- para.φ_l*max.(0, (n[2:para.T-para.init_t+1] - n[1:para.T-para.init_t]))
findall(x -> x < para.c_min, consumption)
plot!(plt, para.init_t:para.T-1, consumption, label = "consumption")
plot!(twinx(), para.init_t:para.T, mort.μ[para.init_t:para.T], label = "mortality", color = :black, lw = 2)