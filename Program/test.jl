include("Mortality.jl")
using .Mortality
include("PensionBenefit.jl")
using .PensionBenefit

using Distributions, LinearAlgebra, CairoMakie, FastGaussQuadrature
using Parameters, Random, Tables, Profile, DataFrames
using CUDA, CUDAKernels, KernelAbstractions, Tullio, CUDA.Adapt
using BenchmarkTools, Interpolations
using JLD2, LaTeXStrings

#life-cycle problem of pension solver
mort = mortality([1.13, 5.0*10^4,0.1,0.0002])
T = life_ceil([1.13, 5.0*10^4,0.1,0.0002])


function benchmark_model(γ = 3.0, β = 1 ./ [1.02, 1.02, 1.02], η = [0.7, 0.9, 1.0], 
                         θ_b = 400, κ = 700, c_floor = 0.3, r = 0.02, φ_l = 5.0,
                         σ = 0.0001, ρ = 1.0, δ = [1.0, 0.0, 0.0], 
                         μ = mort, T = T, reduction = 0.0, sra = 65, t_0 = 40)

    #initialize the model
    asset = range(0, 200, length = 31)
    type = 1:3
    ϵ = range(-2/sqrt(1-ρ^2)*σ, 2/sqrt(1-ρ^2)*σ, length = 5)
    ξ, weight = gausshermite(10)
    ξ = sqrt(2)*σ*ξ
    work = 0:1
    plan = 1:3
    a_prop = 0:0.05:1-1e-5
    aims = range(2.747, 4.58, length = 3)
    wy = 0:30

    v = zeros(length(asset), length(work), length(plan), length(aims), length(ϵ), length(wy), length(type), T-t_0+2)
    a_policy = zeros(length(asset), length(work), length(plan), length(aims), length(ϵ), length(wy), length(type), T-t_0+1)
    n_policy = zeros(length(asset), length(work), length(plan), length(aims), length(ϵ), length(wy), length(type), T-t_0+1)
    p_policy = zeros(length(asset), length(work), length(plan), length(aims), length(ϵ), length(wy), length(type), T-t_0+1)

    return (; # parameters
            γ = γ, β = β, η = η, θ_b = θ_b, κ = κ, c_floor = c_floor, r = r, σ = σ, 
            ρ = ρ, δ = δ, μ = μ, T = T, reduction = reduction, sra = sra, t_0 = t_0, φ_l = φ_l,
            # grids
            asset = asset, work = work, plan = plan, a_prop = a_prop, aims = aims, 
            wy = wy, ϵ = ϵ, ξ = ξ, weight = weight, type = type,
            # cu_grids
            cu_asset = CuArray(asset), cu_work = CuArray(work), cu_plan = CuArray(plan), 
            cu_a_prop = CuArray(a_prop), cu_aims = CuArray(aims), cu_wy = CuArray(wy), 
            cu_ϵ = CuArray(ϵ), cu_ξ = CuArray(ξ), cu_weight = CuArray(weight), cu_type = CuArray(type),
            cu_η = CuArray(η), cu_β = CuArray(β),
            # solution
            v = v, a_policy = a_policy, n_policy = n_policy, p_policy = p_policy)
end
p = benchmark_model()

#=
-------------------------------------------
              Index System
-------------------------------------------
asset       a         aa 
ϵ           e       
ξ           x
prod_factor k                   Not used
rra         d
work        w         ww
aims        m
lms         l                   Not used
plan        p         pp
scheme      c         cc
wy          y
type        b                   
prior       j
inform      i         ii        Not used
-------------------------------------------
=#

function solve_benchmark(;model)
    @unpack γ, β, η, θ_b, κ, c_floor, r, σ, ρ, δ, μ, T, reduction, sra, t_0, φ_l,
            asset, work, plan, a_prop, aims, wy, ϵ, ξ, weight, type,
            cu_asset, cu_work, cu_plan, cu_a_prop, cu_aims, cu_wy, cu_ϵ, cu_ξ, cu_weight, cu_type, cu_η, cu_β, 
            v, a_policy, n_policy, p_policy = model

    # Compute outside the loop
    @tullio leisure[ww,b] := (1 - 260/365*cu_work[ww])^(1-cu_η[b])
    @tullio pension[p,pp,y,m] := ((cu_plan[p] == 1) && (cu_plan[pp] == 2) ? max(cu_wy[y], 2*cu_wy[y]-15) : (cu_plan[pp] == 3 ? max(cu_wy[y]*cu_aims[m]*0.00775+3, cu_wy[y]*cu_aims[m]*0.0155) : 0))
    @tullio adj_cost[w,ww] := $φ_l * (cu_work[w] == 0)*(cu_work[ww] == 1)

    for t in reverse(t_0:T)
        cu_v = cu(linear_interpolation((asset, work, plan, aims, ϵ, wy, type), v[:,:,:,:,:,:,:,t-t_0+2], extrapolation_bc = Line())) 

        mort = μ[t-t_0+1]
        age_factor = δ[1] + δ[2]*t + δ[3]*t^2

        @tullio wage[e,ww] := max(exp(cu_ϵ[e] + $age_factor), 2.747)*cu_work[ww]
        @tullio index_salary[e,ww] := min(4.58, wage[e,ww])*cu_work[ww]
        @tullio tax[e,ww] := 0.2*0.12*index_salary[e]*cu_work[ww] 
        @tullio disposable[a,e,ww,p,pp,y,m,w] := (1+$r)*cu_asset[a] + wage[e,ww] - tax[e,ww] + pension[p,pp,y,m] - adj_cost[w,ww]
        @tullio consumption[a,e,ww,p,pp,y,m,w,ap] := (disposable[a,e,ww,p,pp,y,m,w]*(1-cu_a_prop[ap]))
        @tullio utility[a,e,ww,p,pp,y,m,w,ap,b] := (consumption[a,e,ww,p,pp,y,m,w,ap]^cu_η[b]*leisure[ww,b])^(1-$γ)/(1-$γ)
        
        @tullio new_asset[a,e,ww,p,pp,y,m,w,ap,b,x] := disposable[a,e,ww,p,pp,y,m,w] - consumption[a,e,ww,p,pp,y,m,w,ap] (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), p in 1:length(plan), pp in 1:length(plan), y in 1:length(wy), m in 1:length(aims), w in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))
        @tullio new_work[a,e,ww,p,pp,y,m,w,ap,b,x] := cu_work[ww] (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), p in 1:length(plan), pp in 1:length(plan), y in 1:length(wy), m in 1:length(aims), w in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))
        @tullio new_plan[a,e,ww,p,pp,y,m,w,ap,b,x] := cu_plan[pp] (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), p in 1:length(plan), pp in 1:length(plan), y in 1:length(wy), m in 1:length(aims), w in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))
        @tullio new_aims[a,e,ww,p,pp,y,m,w,ap,b,x] := (cu_work[ww] == 0 ? cu_aims[m] : (cu_wy[y] < 5 ? (cu_aims[m]*cu_wy[y]+index_salary[e,ww])/(cu_wy[y]+1) : (cu_aims[m] < index_salary[e,ww] ? (cu_aims[m]*4+index_salary[e,ww])/5 : cu_aims[m]))) (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), p in 1:length(plan), pp in 1:length(plan), y in 1:length(wy), m in 1:length(aims), w in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))
        @tullio new_ϵ[a,e,ww,p,pp,y,m,w,ap,b,x] := $ρ * cu_ϵ[e] + cu_ξ[x] (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), p in 1:length(plan), pp in 1:length(plan), y in 1:length(wy), m in 1:length(aims), w in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))
        @tullio new_wy[a,e,ww,p,pp,y,m,w,ap,b,x] := min(cu_wy[y] + 1, 30) (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), p in 1:length(plan), pp in 1:length(plan), y in 1:length(wy), m in 1:length(aims), w in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))
        @tullio new_type[a,e,ww,p,pp,y,m,w,ap,b,x] := cu_type[b] (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), p in 1:length(plan), pp in 1:length(plan), y in 1:length(wy), m in 1:length(aims), w in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))

        @tullio bequest[a,e,ww,p,pp,y,m,w,ap,b,x] := $mort*$θ_b*(new_asset[a,e,ww,p,pp,y,m,w,ap,b,x] + $κ)^(1-$γ)/(1-$γ) 

        f_v = (1-mort)*cu_v.(new_asset, new_work, new_plan, new_aims, new_ϵ, new_wy, new_type) .+ mort*bequest
        @tullio EV[a,e,ww,p,pp,y,m,w,ap,b] := f_v[a,e,ww,p,pp,y,m,w,ap,b,x] * cu_weight[x]

        @tullio penalty[p,pp] := (cu_plan[p] > cu_plan[pp])||((cu_plan[p] == 2)&&(cu_plan[pp] == 3))||((abs($t - $sra) > 5)&&(cu_plan[p] != cu_plan[pp])) ? -1e36 : 0

        @tullio candidate[a,w,p,m,e,y,b,ap,ww,pp] := utility[a,e,ww,p,pp,y,m,w,ap,b] + cu_β[b]*EV[a,e,ww,p,pp,y,m,w,ap,b] + penalty[p,pp]

        v[:,:,:,:,:,:,:,t-t_0+1], ind = Array.(dropdims.(findmax(candidate, dims = (8,9,10)), dims = (8,9,10))) 

        a_policy[:,:,:,:,:,:,:,t-t_0+1] = asset[getindex.(ind, 8)]
        n_policy[:,:,:,:,:,:,:,t-t_0+1] = work[getindex.(ind, 9)]
        p_policy[:,:,:,:,:,:,:,t-t_0+1] = plan[getindex.(ind, 10)]
    end
    return (; v = v, a_policy = a_policy, n_policy = n_policy, p_policy = p_policy)
end

solve_benchmark(model = p)

function solve(;model)
    @unpack γ, β, η, θ_b, κ, c_floor, r, σ, ρ, δ, μ, T, reduction, sra, t_0, φ_l,
            asset, work, plan, a_prop, aims, wy, ϵ, ξ, weight, type,
            cu_asset, cu_work, cu_plan, cu_a_prop, cu_aims, cu_wy, cu_ϵ, cu_ξ, cu_weight, cu_type, cu_η, cu_β, 
            v, a_policy, n_policy, p_policy = model

    value = zeros(length(asset), length(ϵ), length(type), T-t_0+2)
    asset_policy = zeros(length(asset), length(ϵ), length(type), T-t_0+1)
    labor_policy = zeros(length(asset), length(ϵ), length(type), T-t_0+1)

    for t in reverse(t_0:T)
        cu_v = cu(LinearInterpolation((asset, ϵ, type), value[:,:,:,t-t_0+2], extrapolation_bc = Line()))
        mort = μ[t]
        age_factor = δ[1] + δ[2]*t + δ[3]*t^2

        @tullio wage[e] := max(exp(cu_ϵ[e] + $age_factor), 2.747)
        @tullio disposable[a,e,ww] := (1+$r)*cu_asset[a] + wage[e]*cu_work[ww] 
        @tullio consumption[a,e,ww,ap] := (disposable[a,e,ww]*(1-cu_a_prop[ap]))
        @tullio utility[a,e,ww,ap,b] := (consumption[a,e,ww,ap]^cu_η[b]*(1-cu_work[ww])^(1-cu_η[b]))^(1-$γ)/(1-$γ) 

        @tullio new_asset[a,e,ww,ap,b,x] := disposable[a,e,ww] - consumption[a,e,ww,ap] (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))
        @tullio new_ϵ[a,e,ww,ap,b,x] := $ρ * cu_ϵ[e] + cu_ξ[x] (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))
        @tullio new_type[a,e,ww,ap,b,x] := cu_type[b] (a in 1:length(asset), e in 1:length(ϵ), ww in 1:length(work), ap in 1:length(a_prop), b in 1:length(type), x in 1:length(ξ))

        new_asset, new_ϵ, new_type = parent.((new_asset, new_ϵ, new_type))
        fv = cu_v.(new_asset, new_ϵ, new_type)
        @tullio EV[a,e,ww,ap,b] := fv[a,e,ww,ap,b,x] * cu_weight[x] 

        @tullio candidate[a,e,b,ap,ww] := utility[a,e,ww,ap,b] + $mort * cu_β[b]*EV[a,e,ww,ap,b]
        candidate = isnan.(candidate) * (-1e36) .+ .!isnan.(candidate) .* candidate
        value[:,:,:,t-t_0+1], ind = Array.(dropdims.(findmax(candidate, dims = (4,5)), dims = (4,5))) 
        asset_policy[:,:,:,t-t_0+1] = a_prop[getindex.(ind, 4)]
        labor_policy[:,:,:,t-t_0+1] = work[getindex.(ind, 5)]
    end
    return (; value = value, asset_policy = asset_policy, labor_policy = labor_policy)
end

sol = solve(model = p)
sol.asset_policy
function simulate(;model, sol)
    @unpack γ, β, η, θ_b, κ, c_floor, r, σ, ρ, δ, μ, T, reduction, sra, t_0, φ_l,
            asset, work, plan, a_prop, aims, wy, ϵ, ξ, weight, type,
            cu_asset, cu_work, cu_plan, cu_a_prop, cu_aims, cu_wy, cu_ϵ, cu_ξ, cu_weight, cu_type, cu_η, cu_β, 
            v, a_policy, n_policy, p_policy = model
    @unpack value, asset_policy, labor_policy = sol

    a_path = 30*ones(T-t_0+1)
    n_path = zeros(T-t_0+1)
    ϵ_path = zeros(T-t_0+1)
    new_ϵ = zeros(T-t_0+1)
    new_a = zeros(T-t_0+1)
    consumption_path = zeros(T-t_0+1)
    wage_path = zeros(T-t_0+1)
    set_type = 3

    for t in t_0:T-1
        s = t - t_0 + 1
        a_func = LinearInterpolation((asset, ϵ, type), asset_policy[:,:,:,s], extrapolation_bc = Line())
        n_func = LinearInterpolation((asset, ϵ, type), labor_policy[:,:,:,s], extrapolation_bc = Line()) 
        wage_path[s] = max(exp(ϵ_path[s] + δ[1] + δ[2]*t + δ[3]*t^2), 2.747)
        n_path[s] = round(clamp(n_func(a_path[s], ϵ_path[s], set_type), 0, 1))
        disposable = (1+r)*a_path[s] + wage_path[s]*n_path[s] 
        new_a[s] = a_func(a_path[s], ϵ_path[s], set_type)*disposable
        consumption_path[s] = disposable - new_a[s]
        a_path[s+1] = new_a[s]
        ϵ_path[s+1] = ρ*ϵ_path[s] + randn()*σ
    end
    return (; a_path = a_path, n_path = n_path, ϵ_path = ϵ_path, consumption_path = consumption_path, wage_path = wage_path, new_a = new_a)
end

sim = simulate(model = p, sol = sol)

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Age")
    xlims!(p.t_0, p.T)
    lines!(ax, p.t_0:p.T, sim.a_path, label = "Asset")
    lines!(ax, p.t_0:p.T, sim.wage_path, label = "Wage")
    lines!(ax, p.t_0:p.T, sim.consumption_path, label = "Consumption")
    vspan!(ax, findall(isone, sim.n_path) .+ p.t_0 .- 1.5, findall(isone, sim.n_path) .+ p.t_0 .- 0.5, color = (:gray, 0.3))
    Legend(fig[1, 2], ax, framevisible = false)
    fig
end