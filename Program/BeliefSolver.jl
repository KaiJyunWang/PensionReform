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
=#

function solve(;BP)
    (; γ, η, r, β, ξ_nodes, ϵ_nodes, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aims, plan, prior, scheme1, scheme2, rra, Q_1, Q_2) = BP
    @assert (abs(ρ) < 1) "|ρ| must be less than 1"
    τ, reduction, sra, fra = zeros(2), zeros(2), zeros(2), zeros(2)
    τ[1], reduction[1], sra[1], fra[1] = scheme1.τ, scheme1.reduction, scheme1.sra, scheme1.fra
    τ[2], reduction[2], sra[2], fra[2] = scheme2.τ, scheme2.reduction, scheme2.sra, scheme2.fra
    # transform to cuda
    τ_cua, reduction_cua, sra_cua, fra_cua, Q_1_cua, Q_2_cua  = cu(τ), cu(reduction), cu(sra), cu(fra), cu(Q_1), cu(Q_2)
    #initialization 
    ξ, weight = gausshermite(ξ_nodes)
    ξ_cua = cu(sqrt(2)*σ*ξ) # remember to devide the integral by sqrt(π)
    weight_cua = cu(weight)
    ϵ = collect(range(-2/sqrt(1-ρ^2)*σ, 2/sqrt(1-ρ^2)*σ, length = ϵ_nodes))
    scheme = [1,2]
    type = [1,2,3]
    # crra 
    u(x) = x^(1-γ)/(1-γ)

    # number of grids
    n_asset, n_ϵ, n_rra, n_work, n_aims, n_plan, n_scheme, n_wy, n_type, n_prior = length(asset), length(ϵ), length(rra), length(work), length(aims), length(plan), length(scheme), length(wy), length(β), length(prior)
    # transform to cuda
    asset_cua, ϵ_cua, rra_cua, work_cua, aims_cua, plan_cua, scheme_cua, wy_cua, β_cua, η_cua, type_cua, prior_cua = cu(asset), cu(ϵ), cu(rra), cu(work), cu(aims), cu(plan), cu(scheme), cu(wy), cu(β), cu(η), cu(type), cu(prior)
    # Float32 vectors
    asset_32, ϵ_32, rra_32, work_32, aims_32, plan_32, scheme_32, wy_32, type_32, prior_32 = Float32.(asset), Float32.(ϵ), Float32.(rra), Float32.(work), Float32.(aims), Float32.(plan), Float32.(scheme), Float32.(wy), Float32.(type), Float32.(prior)

    # value and policy functions
    v = zeros(n_asset, n_ϵ, n_rra, n_work, n_aims, n_plan, n_scheme, n_wy, n_type, n_prior, T-init_t+2)
    #=
    asset_policy = zeros(n_asset, n_ϵ, n_rra, n_work, n_aims, n_plan, n_scheme, n_wy, n_type, n_prior, T-init_t+1)
    work_policy = zeros(n_asset, n_ϵ, n_rra, n_work, n_aims, n_plan, n_scheme, n_wy, n_type, n_prior, T-init_t+1)
    plan_policy = zeros(n_asset, n_ϵ, n_rra, n_work, n_aims, n_plan, n_scheme, n_wy, n_type, n_prior, T-init_t+1)
    =#

    # precomputed matrices 
    @tullio mix_transition[cc,c,j] := Q_1_cua[cc,c]*(1-prior_cua[j]) + Q_2_cua[cc,c]*prior_cua[j] 

    # value function iteration 
    for t in reverse(init_t:T)
        s = t-init_t+1
        println(s)
        
        # Interpolations
        v_func = LinearInterpolation((asset_32, ϵ_32, rra_32, work_32, aims_32, plan_32, scheme_32, wy_32, type_32, prior_32), v[:,:,:,:,:,:,:,:,:,:,s+1], extrapolation_bc=Line())
        cu_v_func = cu(v_func)

        # mortality
        mort = μ[t]

        # factor of production in age
        age_factor = δ[1] + δ[2]*t + δ[3]*t^2
        
        # calculate the utility flow
        @tullio wage[e] := max(exp($age_factor + ϵ_cua[e]), 2.747)
        @tullio index_salary[e] := wage[e] < aims[2] ? aims[1] : (wage[e] < aims[3] ? aims[2] : aims[3])
        @tullio pension[pp,y,m,d,c,p] := (1-reduction_cua[c])*((plan_cua[p] == 1)*(plan_cua[pp] == 2)*max(wy_cua[y], 2*wy_cua[y]-15)*aims_cua[m] + (plan_cua[pp] == 3)*max(wy_cua[y]*aims_cua[m]*0.00775+3, wy_cua[y]*aims_cua[m]*0.0155))*(1+0.04*rra_cua[d])
        @tullio pension_tax[pp,c,e,ww] := (plan_cua[pp]==1)*τ_cua[c]*0.2*index_salary[e]*(work_cua[ww]==1)
        @tullio adj_cost[w,ww] := (work_cua[ww]-work_cua[w]==1)*φ_l
        @tullio consumption[a,e,pp,y,m,d,c,w,aa,ww,p] := (1+$r)*asset_cua[a] + pension[pp,y,m,d,c,p] - pension_tax[pp,c,e,ww] - adj_cost[w,ww] - asset_cua[aa]
        @tullio leisure[ww] := 1 - 260/364*work_cua[ww]
        @tullio utility[a,e,pp,y,m,d,c,w,aa,ww,p] := consumption[a,e,pp,y,m,d,c,w,aa,ww,p] ≥ $c_min ? u(consumption[a,e,pp,y,m,d,c,w,aa,ww,p]^(η_cua[b])*leisure[ww]^(1-η_cua[b])) : -1e38

        # bequest
        @tullio bequest[aa,b] := $mort*β_cua[b]*$θ_b*u($κ + asset_cua[aa])

        # state transitions
        @tullio asset_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := asset_cua[aa] (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)

        @tullio ϵ_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := $ρ*ϵ_cua[e] + ξ_cua[x] (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)
        
        @tullio rra_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := rra_cua[d]*(plan_cua[pp] == plan_cua[p]) + ($t - sra_cua[c])*(plan_cua[pp] > plan_cua[p]) (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)
        
        @tullio work_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := work_cua[ww] (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)
        
        @tullio aims_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := work_cua[ww] == 0 ? aims_cua[m] : (wy_cua[y] < 5 ? (aims_cua[m]*wy_cua[y] + index_salary[e])/(wy_cua[y] + 1) : aims_cua[y] + 0.2*max(0, index_salary[e] - aims_cua[y]) ) (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)
        
        @tullio plan_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := plan_cua[pp] (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)

        @tullio scheme_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := scheme_cua[cc] (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)
        
        @tullio wy_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := min(wy_cua[y] + work_cua[ww], $wy_ceil) (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)
        
        @tullio type_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := type_cua[b] (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)
        
        @tullio prior_next[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] := prior_cua[j] (aa in 1:n_asset, e in 1:n_ϵ, x in 1:ξ_nodes, d in 1:n_rra, pp in 1:n_plan, p in 1:n_plan, c in 1:n_scheme, ww in 1:n_work, m in 1:n_aims, y in 1:n_wy, b in 1:n_type, j in 1:n_prior, cc in 1:n_scheme)

        # EV 
        f_utility = cu_v_func.(asset_next, ϵ_next, rra_next, work_next, aims_next, plan_next, scheme_next, wy_next, type_next, prior_next)
        @tullio E_ϵ[aa,e,d,pp,p,c,ww,m,y,b,j,cc] := weight_cua[x]*f_utility[aa,e,x,d,pp,p,c,ww,m,y,b,j,cc] 
        @tullio EV[aa,e,d,pp,p,c,ww,m,y,b,j] := (1-$mort)/sqrt(π)*E_ϵ[aa,e,d,pp,p,c,ww,m,y,b,j,cc]*mix_transition[cc,c,j] 

        # forbidden paths
        @tullio penalty[p,c,pp] := (plan_cua[pp] < plan_cua[p]) || ((plan_cua[pp] == 3)&&(plan_cua[p] == 2)) || ((plan_cua[pp] != plan_cua[pp])&&(abs($t - sra_cua[c]) > 5)) || ((work[ww] == 1)&&($t > fra_cua[c])) ? -1e38 : 0
        
        @tullio candidate[a,e,d,w,m,p,c,y,b,j,aa,ww,pp] := utility[a,e,p,y,m,d,c,w,aa,ww,p] + β_cua[b]*EV[a,e,d,pp,p,c,ww,m,y,b,j] + bequest[aa,b] + penalty[p,c,pp]

        v[:,:,:,:,:,:,:,:,:,:,s] = dropdims(mapreduce(identity, max, candidate, dims = (11,12,13)), dims = (11,12,13))
    end
    return v
end

BP = @with_kw (γ = 1.5, η = [0.8, 0.9, 1.0], r = 0.04, β = [0.92, 0.95, 0.98], ξ_nodes = 20, ϵ_nodes = 6, T = 80, 
            μ = mort, init_t = 40, ρ = 0.9, σ = 0.2, asset = collect(range(0, 50, length = 21)), work = [0,1], 
            wy = collect(0:30), wy_ceil = 30, c_min = 1e-5, δ = [0.1, 0.01, -0.001], φ_l = 5.0, θ_b = 0.5, κ = 10.0, 
            aims = profile, plan = collect(1:3), prior = collect(0.0:0.2:1.0), 
            scheme1 = ReformScheme(0.12, 0.0, 65, 70), scheme2 = ReformScheme(0.12, 0.2, 65, 70),
            rra = collect(-5:5), Q_1 = [0.9 0.0; 0.1 1.0], Q_2 = [0.5 0.0; 0.5 1.0])
BP = BP()
solve(BP = BP)


