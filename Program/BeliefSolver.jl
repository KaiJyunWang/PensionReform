#-------------------------------------------
# This is the solver and simulation file of a heterogeneous agent model 
# with anticipated reforms. The agents anticipate the reform and adjust 
# their behavior accordingly.
# Execute version 1.9.2.
#-------------------------------------------
include("Mortality.jl")
using .Mortality
include("PensionBenefit.jl")
using .PensionBenefit

using Distributions, LinearAlgebra, CairoMakie, FastGaussQuadrature
using Parameters, Random, Tables, Profile, DataFrames
using CUDA, CUDAKernels, KernelAbstractions, Tullio, CUDA.Adapt
using BenchmarkTools, Interpolations, KernelDensity
using JLD2, LaTeXStrings

# Life-cycle problem of pension solver
mort = mortality([1.5, 1e8, 0.2, 0.0003])
T = life_ceil([1.5, 1e8, 0.2, 0.0003])

# Profile of aime
profile = [2.747, 3.48, 4.58]

function stair(x)
    if x < profile[2]
        return profile[1]
    elseif x < profile[3]
        return profile[2]
    else
        return profile[3]
    end
end

# Solver 
function solve(;para) 
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, lme, prior, scheme1, scheme2) = para
    τ, reduction, sra, fra = zeros(2), zeros(2), zeros(2), zeros(2) 
    τ[1], τ[2] = scheme1.τ, scheme2.τ
    reduction[1], reduction[2] = scheme1.reduction, scheme2.reduction
    sra[1], sra[2] = scheme1.sra, scheme2.sra
    fra[1], fra[2] = scheme1.fra, scheme2.fra
    τ_cua, reduction_cua, sra_cua, fra_cua = cu(τ), cu(reduction), cu(sra), cu(fra)
    #settings
    ξ, w = gausshermite(ξ_nodes)
    n_ξ = length(ξ)
    ξ_cua = cu(ξ)
    w_cua = cu(w)
    n_ϵ = length(ϵ)
    ϵ_grid = zeros(n_ϵ, T-init_t+2)
    ϵ_grid[:,1] = ϵ
    for t in 2:T-init_t+2
        ϵ_grid[:,t] = range(ρ*ϵ_grid[1,t-1] + sqrt(2)*σ*minimum(ξ), ρ*ϵ_grid[end,t-1] + sqrt(2)*σ*maximum(ξ), length(ϵ))
    end
    # relative retire age
    rra = collect(-5:5)
    n_rra = length(rra)
    rra_cua = cu(rra)
    # reform state 
    rs = [1,2] 
    n_rs = length(rs)
    rs_cua = cu(rs)

    #initialization 
    #states
    asset_cua, work_cua, wy_cua, plan_cua, aime_cua, lme_cua, prior_cua, β_cua, η_cua = cu(asset), cu(work), cu(wy), cu(plan), cu(aime), cu(lme), cu(prior), cu(β), cu(η)
    n_asset, n_work, n_wy, n_plan, n_aime, n_lme, n_prior, n_β, n_η = length(asset), length(work), length(wy), length(plan), length(aime), length(lme), length(prior), length(β), length(η)
    @assert n_β == n_η "Lengths of β and η mismatched."

    #value and policy
    v = zeros(n_asset, n_ϵ, n_work, n_wy, n_plan, n_aime, n_lme, n_prior, n_β, n_rra, n_rs, T-init_t+2) 
    policy_asset = zeros(n_asset, n_ϵ, n_work, n_wy, n_plan, n_aime, n_lme, n_prior, n_β, n_rra, n_rs, T-init_t+1) 
    policy_work = zeros(n_asset, n_ϵ, n_work, n_wy, n_plan, n_aime, n_lme, n_prior, n_β, n_rra, n_rs, T-init_t+1) 
    policy_plan = zeros(n_asset, n_ϵ, n_work, n_wy, n_plan, n_aime, n_lme, n_prior, n_β, n_rra, n_rs, T-init_t+1) 

    #retired monthly benefit
    @tullio monthly_benefit[y,m,q,c] := (1-reduction_cua[c])*max(wy_cua[m]*aime_cua[y]*0.00775+3, wy_cua[m]*aime_cua[y]*0.0155)*(1+0.04*(rra_cua[q]))
    #lump-sum benefit
    @tullio lumpsum_benefit[y,m,q,c] := (1-reduction_cua[c])*min(max(wy_cua[m], 2*wy_cua[m]-15), 50)*aime_cua[y]*(1+0.04*(rra_cua[q]))
    #adjustment cost of working
    @tullio adj_cost[k,x] := $φ_l*(work_cua[k]-work_cua[x] == 1)
    #bequest
    @tullio bequest[b,j] := β_cua[b]*($θ_b*($κ+asset_cua[j])^(1-$γ))/(1-$γ)

    #value function iteration
    for s in 1:T-init_t+1
        t = T-s+1 
        mort = μ[t] 
        wy_comp = δ[1] + δ[2]*t + δ[3]*t^2 
        v_func = LinearInterpolation((asset, ϵ_grid[:,t-init_t+2], work, wy, plan, aime, lme, prior, β, rra, rs), v[:,:,:,:,:,:,:,:,:,:,:,t-init_t+2])
        v_func_cua = cu(v_func)
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        
        #minimum wage
        @tullio wage[l] := max(2.747, exp(ϵ_cua[l] + $wy_comp)) 
        @tullio force_retire[l,c] := (exp(ϵ_cua[l] + $wy_comp) < 2.747)&&($t ≥ fra[c])
        #current wage transformed into aime
        @tullio c_aime[l] := (wage[l] < aime[2] ? aime[1] : (wage[l] < aime[3] ? aime[2] : aime[3])) 
        @tullio pension_tax[l] := c_aime[l]*0.2*τ[c]
        @tullio consumption[i,j,k,l,m,x,y,p,c] := (1+$r)*asset_cua[i] + work_cua[k]*wage[l] - adj_cost[k,x] - pension_tax[l]*work_cua[k]*(plan_cua[p] == 1) + (plan_cua[p] == 2)*lumpsum_benefit[y,m,q,c] + (wy_cua[m]≥15)*(plan_cua[p]==3)*monthly_benefit[y,m,q,c] - asset_cua[j] 
        @tullio utility[i,j,k,l,m,x,y,p,c,b] := consumption[i,j,k,l,m,x,y,p,c] ≥ $c_min ? ((consumption[i,j,k,l,m,x,y,p,c]^η_cua[b]*(1-264/360*work_cua[k])^(1-η_cua[b]))^(1-$γ))/(1-$γ) : -1e38 
        
        # law of mortion
        @tullio f_ϵ[j,l,k,m,p,y,z,a,b,q,h,c] := $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ_cua[h] (j in 1:n_asset, l in 1:n_ϵ, k in 1:n_work, m in 1:n_wy, p in 1:n_plan, y in 1:n_aime, z in 1:n_lme, a in 1:n_prior, b in 1:n_β, q in 1:n_rra, h in 1:n_ξ, c in 1:n_rs) 
        @tullio f_asset[j,k,l,m,p,y,z,a,b,q,h,c] := asset_cua[j] (j in 1:n_asset, l in 1:n_ϵ, k in 1:n_work, m in 1:n_wy, p in 1:n_plan, y in 1:n_aime, z in 1:n_lme, a in 1:n_prior, b in 1:n_β, q in 1:n_rra, h in 1:n_ξ, c in 1:n_rs)  
        @tullio f_work[j,k,l,m,p,y,z,a,b,q,h,c] := work_cua[k] (j in 1:n_asset, l in 1:n_ϵ, k in 1:n_work, m in 1:n_wy, p in 1:n_plan, y in 1:n_aime, z in 1:n_lme, a in 1:n_prior, b in 1:n_β, q in 1:n_rra, h in 1:n_ξ, c in 1:n_rs) 
        @tullio f_wy[j,k,l,m,p,y,z,a,b,q,h,c] := min(wy_cua[m]+work_cua[k], $wy_ceil) (j in 1:n_asset, l in 1:n_ϵ, k in 1:n_work, m in 1:n_wy, p in 1:n_plan, y in 1:n_aime, z in 1:n_lme, a in 1:n_prior, b in 1:n_β, q in 1:n_rra, h in 1:n_ξ, c in 1:n_rs) 
        @tullio f_plan[j,k,l,m,p,y,z,a,b,q,h,c] := plan_cua[p] (j in 1:n_asset, l in 1:n_ϵ, k in 1:n_work, m in 1:n_wy, p in 1:n_plan, y in 1:n_aime, z in 1:n_lme, a in 1:n_prior, b in 1:n_β, q in 1:n_rra, h in 1:n_ξ, c in 1:n_rs)
    end
end

#parameters 
RS_1 = ReformScheme(0.12, 0.0, 70, 65)
RS_2 = ReformScheme(0.12, 0.0, 65, 65)
BP = @with_kw (γ = 3.0, η = [0.5, 0.7, 0.9], r = 0.02, β = [0.92, 0.98, 1.02], ξ_nodes = 20, ϵ = range(0.0, 3.0, 5),
    T = T, μ = mort, init_t = 40, ρ = 0.97, σ = 0.06, asset = collect(range(0.0, 30.0, 21)), 
    work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.1, δ = [-1.0, 0.06, -0.0006], φ_l = 5.0, θ_b = 5.0, κ = 2.0,
    aime = profile, plan = collect(1:3), lme = profile, prior = collect(0.0:0.1:1.0), scheme1 = RS_1, scheme2 = RS_2)
BP = BP() 

solve(para = BP)

