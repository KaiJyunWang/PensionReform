#-------------------------------------------
# This is the solver and simulation file of a heterogeneous agent model 
# with belief update. 
# The agents anticipate the reform. 
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

#life-cycle problem of pension solver
mort = mortality([1.5, 1e8, 0.2, 0.0003])
T = life_ceil([1.5, 1e8, 0.2, 0.0003])

#profile of aime
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

function solve(;para)
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, lme, prior, scheme1, scheme2) = para
    τ, reduction, sra, fra = zeros(2), zeros(2), zeros(2), zeros(2)
    τ[1], reduction[1], sra[1], fra[1] = scheme1.τ, scheme1.reduction, scheme1.sra, scheme1.fra
    τ[2], reduction[2], sra[2], fra[2] = scheme2.τ, scheme2.reduction, scheme2.sra, scheme2.fra
    τ_cua, reduction_cua, sra_cua, fra_cua = cu(τ), cu(reduction), cu(sra), cu(fra)
    #initialization 
    ξ, w = gausshermite(ξ_nodes)
    ξ_cua = cu(ξ)
    w_cua = cu(w)
    ϵ_grid = zeros(length(ϵ), T-init_t+2)
    ϵ_grid[:,1] = ϵ
    for t in 2:T-init_t+2
        ϵ_grid[:,t] = range(ρ*ϵ_grid[1,t-1] + sqrt(2)*σ*minimum(ξ), ρ*ϵ_grid[end,t-1] + sqrt(2)*σ*maximum(ξ), length(ϵ))
    end
    rra = collect(-5:5) 
    rra_cua = cu(rra)
    scheme = [1, 2]
    # number of grids
    n_asset, n_ϵ, n_rra, n_work, n_aime, n_lme, n_plan, n_scheme, n_wy, n_type, n_prior = length(asset), length(ϵ), length(rra), length(work), length(aime), length(lme), length(plan), 2, length(wy), length(β), length(prior)
    # transform to cuda
    asset_cua, work_cua, aime_cua, lme_cua, plan_cua, scheme_cua, wy_cua, β_cua, η_cua, prior_cua = cu(asset), cu(work), cu(aime), cu(lme), cu(plan), cu(scheme), cu(wy), cu(β), cu(η), cu(prior) 

    # value and policy functions
    v = zeros(n_asset, n_ϵ, n_rra, n_work, n_aime, n_lme, n_plan, n_scheme, n_wy, n_type, n_prior, T-init_t+2)
    asset_policy = zeros(n_asset, n_ϵ, n_rra, n_work, n_aime, n_lme, n_plan, n_scheme, n_wy, n_type, n_prior, T-init_t+1)
    work_policy = zeros(n_asset, n_ϵ, n_rra, n_work, n_aime, n_lme, n_plan, n_scheme, n_wy, n_type, n_prior, T-init_t+1)
    aime_policy = zeros(n_asset, n_ϵ, n_rra, n_work, n_aime, n_lme, n_plan, n_scheme, n_wy, n_type, n_prior, T-init_t+1)

    # value function iteration
    for t in reverse(init_t:T)
        s = t-init_t+1
        mort = μ[t]
        hc = δ[1] + δ[2]*t + δ[3]*t^2 
        ϵ_cua = cu(ϵ_grid[:,s])
        
        v_func = LinearInterpolation((asset, ϵ_grid[:, s+1], rra, work, aime, lme, plan, scheme, wy, β, prior), v[:,:,:,:,:,:,:,:,:,:,:,s+1])
        cu_v_func = cu(v_func)

        # compute current utility
        @tullio wage[l] := max(2.747, exp(ϵ_cua[l] + $wy_comp))
        @tullio c_aime[l] := (wage[l] < aime[2] ? aime[1] : (wage[l] < aime[3] ? aime[2] : aime[3])) 
        @tullio pension_tax[l] := c_aime[l]*0.2*$τ 
        @tullio pension[p,d,m,y,q] :=  reduction_cua[d]*max(wy_cua[m]*aime_cua[y]*0.00775+3, wy_cua[m]*aime_cua[y]*0.0155)*(1+0.04*rra_cua[q])*(plan[p] == 3) + reduction_cua[d]*min(max(wy_cua[m], 2*wy_cua[m]-15), 50)*aime_cua[y]*(1+0.04*rra_cua[q])*(plan[p] == 2)
        @tullio adj_cost[k,x] := $φ_l*(work_cua[k]-work_cua[x] == 1)
        @tullio bequest[b,j] := β_cua[b]*($θ_b*($κ+asset_cua[j])^(1-$γ))/(1-$γ)
        @tullio consumption[i,j,l,m,k,x,y,p,d,q] := (1+$r)*asset_cua[i] + (wage[l]-pension_tax[l])*work_cua[k] - adj_cost[k,x] - asset_cua[j] + pension[p,d,y,m,q] 
        @tullio utility[i,j,l,m,k,x,y,p,d,q,b] := consumption[i,j,l,m,k,x,y,p,d,q] ≥ $c_min ? (consumption[i,j,l,m,k,x,y,p,d,q]^η_cua[b]*(1-264/360*work_cua[k])^(1-η_cua[b]))^(1-$γ)/(1-$γ) : -1e38

        # state transition
        @tullio asset_next[j,l,q,k,y,z,p,d,m,b,a,h] := asset_cua[j] (j in 1:n_asset, l in 1:n_ϵ, q in 1:n_rra, k in 1:n_work, y in 1:n_aime, z in 1:n_lme, p in 1:n_plan, d in 1:n_scheme, m in 1:n_wy, b in 1:n_type, a in 1:n_prior, h in 1:ξ_nodes)
        @tullio work_next[j,l,q,k,y,z,p,d,m,b,a,h] := work_cua[k] (j in 1:n_asset, l in 1:n_ϵ, q in 1:n_rra, k in 1:n_work, y in 1:n_aime, z in 1:n_lme, p in 1:n_plan, d in 1:n_scheme, m in 1:n_wy, b in 1:n_type, a in 1:n_prior, h in 1:ξ_nodes)
        @tullio ϵ_next[j,l,q,k,y,z,p,d,m,b,a,h] := $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ_cua[h] (j in 1:n_asset, l in 1:n_ϵ, q in 1:n_rra, k in 1:n_work, y in 1:n_aime, z in 1:n_lme, p in 1:n_plan, d in 1:n_scheme, m in 1:n_wy, b in 1:n_type, a in 1:n_prior, h in 1:ξ_nodes)
        @tullio rra_next[j,l,q,k,y,z,p,d,m,b,a,h] := rra[q] (j in 1:n_asset, l in 1:n_ϵ, q in 1:n_rra, k in 1:n_work, y in 1:n_aime, z in 1:n_lme, p in 1:n_plan, d in 1:n_scheme, m in 1:n_wy, b in 1:n_type, a in 1:n_prior, h in 1:ξ_nodes)
        @tullio less_5y_f_aime[y,l,m,k] := c_aime[l]*work_cua[k]/(wy_cua[m]+work_cua[k]) + wy_cua[m]/(wy_cua[m]+work_cua[k])*aime_cua[y]
        @tullio more_5y_f_aime[y,z,l,k] := aime_cua[y] + 0.2*max(0, c_aime[l]*work_cua[k]-lme_cua[z])
        @tullio f_aime[y,z,l,m,k] := (((wy_cua[m] + work_cua[k]) > 0) ? (wy_cua[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,k]) : 2.747)
        @tullio aime_next[j,l,q,k,y,z,p,d,m,b,a,h] := min(4.58, f_aime[y,z,l,m,k]) (j in 1:n_asset, l in 1:n_ϵ, q in 1:n_rra, k in 1:n_work, y in 1:n_aime, z in 1:n_lme, p in 1:n_plan, d in 1:n_scheme, m in 1:n_wy, b in 1:n_type, a in 1:n_prior, h in 1:ξ_nodes)
        @tullio lme_next[j,l,q,k,y,z,p,d,m,b,a,h] := min(4.58, max(lme_cua[z], wage[l]*work_cua[k])) (j in 1:n_asset, l in 1:n_ϵ, q in 1:n_rra, k in 1:n_work, y in 1:n_aime, z in 1:n_lme, p in 1:n_plan, d in 1:n_scheme, m in 1:n_wy, b in 1:n_type, a in 1:n_prior, h in 1:ξ_nodes)
        @tullio plan_next[j,l,q,k,y,z,p,d,m,b,a,h] := plan_cua[p] (j in 1:n_asset, l in 1:n_ϵ, q in 1:n_rra, k in 1:n_work, y in 1:n_aime, z in 1:n_lme, p in 1:n_plan, d in 1:n_scheme, m in 1:n_wy, b in 1:n_type, a in 1:n_prior, h in 1:ξ_nodes)
        
    end 
end

