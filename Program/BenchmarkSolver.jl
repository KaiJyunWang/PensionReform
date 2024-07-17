#-------------------------------------------
# This is the solver and simulattion file of a benchmark model. 
# The agents do not anticipate the reform. 
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
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra) = para
    #settings
    ξ, w = gausshermite(ξ_nodes)
    ξ_cua = cu(ξ)
    w_cua = cu(w)
    ϵ_grid = zeros(length(ϵ), T-init_t+2)
    ϵ_grid[:,1] = ϵ
    for t in 2:T-init_t+2
        ϵ_grid[:,t] = range(ρ*ϵ_grid[1,t-1] + sqrt(2)*σ*minimum(ξ), ρ*ϵ_grid[end,t-1] + sqrt(2)*σ*maximum(ξ), length(ϵ))
    end
    real_retire_age = cu(collect(-5:5))

    #policy functions 
    policy_retire_with_pension = zeros(length(asset), length(aime), length(wy), length(real_retire_age), T-ra+5)
    policy_retire_no_pension = zeros(length(asset), T-ra+5)
    policy_window = zeros(length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(real_retire_age), 3)
    policy_before_window = zeros(length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(init_t:ra-6), 2)

    #value functions
    v_retire_with_pension = zeros(length(asset), length(aime), length(wy), length(real_retire_age), T-ra+6)
    v_retire_no_pension = zeros(length(asset), T-ra+6) 
    v_window = zeros(length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(real_retire_age))
    v_before_window = zeros(length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(init_t:ra-6)) 

    #states
    asset_cua, work_cua, wy_cua, plan_cua, aime_cua, lme_cua = cu(asset), cu(work), cu(wy), cu(plan), cu(aime), cu(lme)
    
    f_v_work = zeros(length(asset), length(work), length(ϵ), length(wy), length(aime), length(lme), ξ_nodes) |> x -> CuArray(x)
    #pre-computing
    #retired monthly benefit
    @tullio monthly_benefit[y,m,q] := max(wy_cua[m]*aime[y]*0.00775+3, wy_cua[m]*aime[y]*0.0155)*(1+0.04*(real_retire_age[q]))
    #lump-sum benefit
    @tullio lumpsum_benefit[y,m,q] := min(max(wy_cua[m], 2*wy_cua[m]-15), 50)*aime[y]*(1+0.04*(real_retire_age[q]))
    #adjustment cost of working
    @tullio adj_cost[k,x] := $φ_l*(work_cua[k]-work_cua[x] == 1)
    #bequest
    @tullio bequest[j] := $β*($θ_b*($κ+asset_cua[j])^(1-$γ))/(1-$γ)
    #future work years
    @tullio f_wy[m,k] := min(wy_cua[m]+work_cua[k], $wy_ceil)

    
    #VFI for retired with pension
    for s in 1:T-ra+5
        t = T-s+1
        mort = μ[t]
        new_vr = v_retire_with_pension[:,:,:,:,end-s+1] |> x -> CuArray(x)
        
        @tullio retired_with_pension_consumption[i,j,y,m,q] := (1+$r)*asset_cua[i] + monthly_benefit[y,m,q] - asset_cua[j]
        @tullio retired_with_pension_utility[i,j,y,m,q] := retired_with_pension_consumption[i,j,y,m,q] ≥ $c_min ? retired_with_pension_consumption[i,j,y,m,q]^((1-$γ)*$η)/(1-$γ) : -1e38
        @tullio candidate_retire_with_pension[i,y,m,q,j] := retired_with_pension_utility[i,j,y,m,q] + (1-$mort)*$β*new_vr[j,y,m,q] + bequest[j]*$mort

        can_retire_with_pension = Array(candidate_retire_with_pension)

        for state in CartesianIndices(v_retire_with_pension[:,:,:,:,T-ra+5-s+1])
            v_retire_with_pension[state, T-ra+5-s+1], ind = findmax(can_retire_with_pension[state,:])
            policy_retire_with_pension[state, T-ra+5-s+1] = asset[ind]
        end
    end
    
    #VFI for retired with no pension
    for s in 1:T-ra+5
        t = T-s+1
        mort = μ[t]
        new_vr = v_retire_no_pension[:,end-s+1] |> x -> CuArray(x)
        
        @tullio retired_no_pension_consumption[i,j] := (1+$r)*asset_cua[i] - asset_cua[j]
        @tullio retired_no_pension_utility[i,j] := retired_no_pension_consumption[i,j] ≥ $c_min ? retired_no_pension_consumption[i,j]^((1-$γ)*$η)/(1-$γ) : -1e38
        @tullio candidate_retire_no_pension[i,j] := retired_no_pension_utility[i,j] + (1-$mort)*$β*new_vr[j] + bequest[j]*$mort

        can_retire_no_pension = Array(candidate_retire_no_pension)

        for state in CartesianIndices(v_retire_no_pension[:,T-ra+5-s+1])
            v_retire_no_pension[state, T-ra+5-s+1], ind = findmax(can_retire_no_pension[state,:])
            policy_retire_no_pension[state, T-ra+5-s+1] = asset[ind]
        end
    end
    
    #VFI for pension window
    for s in 1:length(real_retire_age)
        t = ra+6-s
        wy_comp = δ[1] + δ[2]*t + δ[3]*t^2
        mort = μ[t]
        if s != 1
            v_func = LinearInterpolation((Float32.(asset), Float32.(ϵ_grid[:,t-init_t+2]), Float32.(wy), Float32.(work), Float32.(aime), Float32.(lme)), v_window[:,:,:,:,:,:,end-s+2])
            cu_v_func = cu(v_func)
        end
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        v_retire_no_pension_cua = CuArray(v_retire_no_pension[:,length(real_retire_age)-s+2])
        v_retire_with_pension_cua = CuArray(v_retire_with_pension[:,:,:,length(real_retire_age)-s+1,length(real_retire_age)-s+2])
        
        #future ϵ, avg. 
        @tullio f_ϵ[l,h] := $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ_cua[h]
        #minimum wage is 2.747
        @tullio wage[l] := max(2.747, exp(ϵ_cua[l] + $wy_comp))
        @tullio force_retire[l] := (exp(ϵ_cua[l] + $wy_comp) < 2.747)&&($t ≥ $fra)
        #current wage transformed into aime
        @tullio c_aime[l] := (wage[l] < aime[2] ? aime[1] : (wage[l] < aime[3] ? aime[2] : aime[3])) 
        @tullio pension_tax[l] := c_aime[l]*0.2*$τ
        @tullio window_consumption[i,j,l,m,k,x,q,y] := (1+$r)*asset_cua[i] + wage[l]*work_cua[k] + (plan_cua[q] == 2)*lumpsum_benefit[y,m,12-$s] + (wy_cua[m]≥15)*(plan_cua[q]==3)*monthly_benefit[y,m,12-$s] - pension_tax[l]*work_cua[k]*(plan_cua[q] == 1) - adj_cost[k,x] - asset_cua[j]
        @tullio window_utility[i,j,l,m,k,x,q,y] := (window_consumption[i,j,l,m,k,x,q,y] ≥ $c_min ? (((window_consumption[i,j,l,m,k,x,q,y]^($η))*((1-260/364*work_cua[k])^(1-$η)))^(1-$γ))/(1-$γ) : -1e38)
        #future lowest monthly wage
        @tullio f_lme[z,l,k] := min(4.58, max(lme_cua[z], wage[l]*work_cua[k]))
        #future aime
        @tullio less_5y_f_aime[y,l,m,k] := c_aime[l]*work_cua[k]/(wy_cua[m]+work_cua[k]) + wy_cua[m]/(wy_cua[m]+work_cua[k])*aime_cua[y]
        @tullio more_5y_f_aime[y,z,l,k] := aime_cua[y] + 0.2*max(0, c_aime[l]*work_cua[k]-lme_cua[z])
        @tullio f_aime[y,z,l,m,k] := (((wy_cua[m] + work_cua[k]) > 0) ? (wy_cua[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,k]) : 2.747)
        @tullio f_aime[y,z,l,m,k] = min(4.58, f_aime[y,z,l,m,k])
        if s == 1
            #70 must be the last year of working
            @tullio f_utility[j,k,l,m,y,z,q,h] := (plan_cua[q] == 1 ? -1e38 : ((plan_cua[q] == 3)&&(wy_cua[m]≥15) ? v_retire_with_pension_cua[j,y,m] : v_retire_no_pension_cua[j])) (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), q in 1:length(plan), h in 1:ξ_nodes)
        else
            #for itp purpose
            
            @tullio f_asset_s[j,k,l,m,y,z,h] := asset_cua[j] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_ϵ_s[j,k,l,m,y,z,h] := f_ϵ[l,h] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_wy_s[j,k,l,m,y,z,h] := f_wy[m,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_work_s[j,k,l,m,y,z,h] := work_cua[k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_aime_s[j,k,l,m,y,z,h] := f_aime[y,z,l,m,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_lme_s[j,k,l,m,y,z,h] := f_lme[z,l,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = parent(f_asset_s), parent(f_ϵ_s), parent(f_wy_s), parent(f_work_s), parent(f_aime_s), parent(f_lme_s)
            f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = Float32.(f_asset_s), Float32.(f_ϵ_s), Float32.(f_wy_s), Float32.(f_work_s), Float32.(f_aime_s), Float32.(f_lme_s)
            
            f_v_work = cu_v_func.(f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s)
            @tullio f_utility[j,k,l,m,y,z,q,h] := ((force_retire[l] == 0)&&(plan_cua[q] == 1) ? f_v_work[j,k,l,m,y,z,h] : ((plan_cua[q] == 3)&&(wy_cua[m]≥15) ? v_retire_with_pension_cua[j,y,m] : v_retire_no_pension_cua[j]))
        end
        #EV
        @tullio EV[j,k,l,m,y,z,q] := w_cua[h]*f_utility[j,k,l,m,y,z,q,h]

        @tullio candidate_window[i,l,m,x,y,z,j,k,q] := ((($t == $ra + 5)&&(plan_cua[q] == 1))||((force_retire[l] == 0)&&(work_cua[k] == 1)) ? -1e38 : window_utility[i,j,l,m,k,x,q,y] + (1-$mort)*$β*EV[j,k,l,m,y,z,q] + $mort*bequest[j])

        can_window = Array(candidate_window)

        for state in CartesianIndices(v_window[:,:,:,:,:,:,length(real_retire_age)-s+1])
            v_window[state, length(real_retire_age)-s+1], ind = findmax(can_window[state,:,:,:])
            policy_window[state, length(real_retire_age)-s+1,1] = asset[ind[1]]
            policy_window[state, length(real_retire_age)-s+1,2] = work[ind[2]]
            policy_window[state, length(real_retire_age)-s+1,3] = plan[ind[3]]
        end
    end
    
    #VFI for before window
    #last period before window
    t = ra-6
    mort = μ[t]
    wy_comp = δ[1] + δ[2]*t + δ[3]*t^2
    v_func = LinearInterpolation((Float32.(asset), Float32.(ϵ_grid[:,t-init_t+2]), Float32.(wy), Float32.(work), Float32.(aime), Float32.(lme)), v_window[:,:,:,:,:,:,1])
    cu_v_func = cu(v_func)
    ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
    #future ϵ, avg. 
    @tullio f_ϵ[l,h] := $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ_cua[h]
    #minimum wage is 2.747
    @tullio wage[l] := max(2.747, exp(ϵ_cua[l] + $wy_comp))
    #current wage transformed into aime
    @tullio c_aime[l] := (wage[l] < aime[2] ? aime[1] : (wage[l] < aime[3] ? aime[2] : aime[3]))
    @tullio pension_tax[l] := c_aime[l]*0.2*$τ
    @tullio before_window_consumption[i,j,l,k,x] := (1+$r)*asset_cua[i] + wage[l]*work_cua[k] - pension_tax[l]*work_cua[k] - adj_cost[k,x] - asset_cua[j]
    @tullio before_window_utility[i,j,l,k,x] := before_window_consumption[i,j,l,k,x] ≥ $c_min ? (((before_window_consumption[i,j,l,k,x]^($η))*((1-260/364*work_cua[k])^(1-$η)))^(1-$γ))/(1-$γ) : -1e38
    #future lowest monthly wage
    @tullio f_lme[z,l,k] := min(4.58, max(lme_cua[z], wage[l]*work_cua[k]))
    #future aime
    @tullio less_5y_f_aime[y,l,m,k] := c_aime[l]*work_cua[k]/(wy_cua[m]+work_cua[k]) + wy_cua[m]/(wy_cua[m]+work_cua[k])*aime_cua[y]
    @tullio more_5y_f_aime[y,z,l,k] := aime_cua[y] + 0.2*max(0, c_aime[l]*work_cua[k]-lme_cua[z])
    @tullio f_aime[y,z,l,m,k] := (((wy_cua[m] + work_cua[k]) > 0) ? (wy_cua[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,k]) : 2.747)
    @tullio f_aime[y,z,l,m,k] = min(4.58, f_aime[y,z,l,m,k])

    @tullio f_asset_s[j,k,l,m,y,z,h] := asset_cua[j] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_ϵ_s[j,k,l,m,y,z,h] := f_ϵ[l,h] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_wy_s[j,k,l,m,y,z,h] := f_wy[m,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_work_s[j,k,l,m,y,z,h] := work_cua[k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_aime_s[j,k,l,m,y,z,h] := f_aime[y,z,l,m,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_lme_s[j,k,l,m,y,z,h] := f_lme[z,l,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = parent(f_asset_s), parent(f_ϵ_s), parent(f_wy_s), parent(f_work_s), parent(f_aime_s), parent(f_lme_s)
    f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = Float32.(f_asset_s), Float32.(f_ϵ_s), Float32.(f_wy_s), Float32.(f_work_s), Float32.(f_aime_s), Float32.(f_lme_s)

    f_utility_before_window = cu_v_func.(f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s)
    
    #EV
    @tullio EV_before_window[j,k,l,m,y,z] := w_cua[h]*f_utility_before_window[j,k,l,m,y,z,h]
    @tullio candidate_before_window[i,l,m,x,y,z,j,k] := before_window_utility[i,j,l,k,x] + (1-$mort)*$β*EV_before_window[j,k,l,m,y,z] + $mort*bequest[j] 

    can_before_window = Array(candidate_before_window)

    for state in CartesianIndices(v_before_window[:,:,:,:,:,:,t-init_t+1])
        v_before_window[state, t-init_t+1], ind = findmax(can_before_window[state,:,:])
        policy_before_window[state, t-init_t+1,1] = asset[ind[1]]
        policy_before_window[state, t-init_t+1,2] = work[ind[2]]
    end
    
    for s in 2:ra-6-init_t+1
        t = ra-5-s
        mort = μ[t]
        wy_comp = δ[1] + δ[2]*t + δ[3]*t^2
        v_func = LinearInterpolation((Float32.(asset), Float32.(ϵ_grid[:,t-init_t+2]), Float32.(wy), Float32.(work), Float32.(aime), Float32.(lme)), v_before_window[:,:,:,:,:,:,end-s+2])
        cu_v_func = cu(v_func)
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        #future ϵ, avg. 
        @tullio f_ϵ[l,h] := $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ_cua[h]
        #minimum wage is 2.747
        @tullio wage[l] := max(2.747, exp(ϵ_cua[l] + $wy_comp))
        #current wage transformed into aime
        @tullio c_aime[l] := (wage[l] < aime[2] ? aime[1] : (wage[l] < aime[3] ? aime[2] : aime[3]))
        @tullio pension_tax[l] := c_aime[l]*0.2*$τ
        @tullio before_window_consumption[i,j,l,k,x] := (1+$r)*asset_cua[i] + wage[l]*work_cua[k] - pension_tax[l]*work_cua[k] - adj_cost[k,x] - asset_cua[j]
        @tullio before_window_utility[i,j,l,k,x] := before_window_consumption[i,j,l,k,x] ≥ $c_min ? (((before_window_consumption[i,j,l,k,x]^($η))*((1-260/364*work_cua[k])^(1-$η)))^(1-$γ))/(1-$γ) : -1e38
        #future lowest monthly wage
        @tullio f_lme[z,l,k] := min(4.58, max(lme_cua[z], wage[l]*work_cua[k]))
        #future aime
        @tullio less_5y_f_aime[y,l,m,k] := c_aime[l]*work_cua[k]/(wy_cua[m]+work_cua[k]) + wy_cua[m]/(wy_cua[m]+work_cua[k])*aime_cua[y]
        @tullio more_5y_f_aime[y,z,l,k] := aime_cua[y] + 0.2*max(0, c_aime[l]*work_cua[k]-lme[z])
        @tullio f_aime[y,z,l,m,k] := (((wy_cua[m] + work_cua[k]) > 0) ? (wy_cua[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,k]) : 2.747)
        @tullio f_aime[y,z,l,m,k] = min(4.58, f_aime[y,z,l,m,k])

        @tullio f_asset_s[j,k,l,m,y,z,h] := asset_cua[j] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_ϵ_s[j,k,l,m,y,z,h] := f_ϵ[l,h] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_wy_s[j,k,l,m,y,z,h] := f_wy[m,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_work_s[j,k,l,m,y,z,h] := work_cua[k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_aime_s[j,k,l,m,y,z,h] := f_aime[y,z,l,m,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_lme_s[j,k,l,m,y,z,h] := f_lme[z,l,k] (j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = parent(f_asset_s), parent(f_ϵ_s), parent(f_wy_s), parent(f_work_s), parent(f_aime_s), parent(f_lme_s)
        f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = Float32.(f_asset_s), Float32.(f_ϵ_s), Float32.(f_wy_s), Float32.(f_work_s), Float32.(f_aime_s), Float32.(f_lme_s)

        CUDA.@sync f_utility_before_window = cu_v_func.(f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s)
        #EV
        @tullio EV_before_window[j,k,l,m,y,z] = w_cua[h]*f_utility_before_window[j,k,l,m,y,z,h]
        @tullio candidate_before_window[i,l,m,x,y,z,j,k] = before_window_utility[i,j,l,k,x] + (1-$mort)*$β*EV_before_window[j,k,l,m,y,z] + $mort*bequest[j] 

        can_before_window = Array(candidate_before_window)

        for state in CartesianIndices(v_before_window[:,:,:,:,:,:,t-init_t+1])
            v_before_window[state, t-init_t+1], ind = findmax(can_before_window[state,:,:])
            policy_before_window[state, t-init_t+1,1] = asset[ind[1]]
            policy_before_window[state, t-init_t+1,2] = work[ind[2]]
        end
    end
    
    
    raw_solution = @with_kw (ϵ_grid = ϵ_grid, policy_before_window = Array(policy_before_window), policy_window = Array(policy_window), policy_retire_no_pension = Array(policy_retire_no_pension), policy_retire_with_pension = Array(policy_retire_with_pension), v_before_window = Array(v_before_window), v_window = Array(v_window), v_retire_no_pension = Array(v_retire_no_pension), v_retire_with_pension = Array(v_retire_with_pension))
    return raw_solution()
end 

function integrate_sol(sol;para)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra) = para
    (; ϵ_grid, policy_before_window, policy_window, policy_retire_no_pension, policy_retire_with_pension, v_before_window, v_window, v_retire_no_pension, v_retire_with_pension) = sol
    real_retire_age = collect(-5:5)

    policy_plan = ones(length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(real_retire_age), T-init_t+1) 
    policy_asset = zeros(size(policy_plan))
    policy_work = zeros(size(policy_plan))
    v = zeros(size(policy_plan))
    policy_plan[:,:,:,:,:,:,:,ra-5-init_t+1:ra+5-init_t+1] = stack(fill(policy_window[:,:,:,:,:,:,:,3],11))
    
    for t in 2:T-init_t+1
        @tullio policy_plan[i,l,m,x,y,z,q,$t] = policy_plan[i,l,m,x,y,z,q,$t-1] != 1 ? policy_plan[i,l,m,x,y,z,q,$t-1] : policy_plan[i,l,m,x,y,z,q,$t]
    end
    
    for s in 1:ra-6-init_t+1
        policy_asset[:,:,:,:,:,:,:,s] = stack(fill(policy_before_window[:,:,:,:,:,:,s,1],11))
        policy_work[:,:,:,:,:,:,:,s] = stack(fill(policy_before_window[:,:,:,:,:,:,s,2],11))
        v[:,:,:,:,:,:,:,s] = stack(fill(v_before_window[:,:,:,:,:,:,s],11))
    end

    for s in ra-6-init_t+2:ra+5-init_t+1
        @tullio policy_asset[i,l,m,x,y,z,q,$s] = (policy_plan[i,l,m,x,y,z,q,$s-1] == 1 ? policy_window[i,l,m,x,y,z,$s-$ra+6+$init_t-1,1] : (policy_plan[i,l,m,x,y,z,q,$s-1] == 2 ? policy_retire_no_pension[i,$s-$ra+6+$init_t-2] : policy_retire_with_pension[i,y,m,q,$s-$ra+6+$init_t-2]))
        @tullio policy_work[i,l,m,x,y,z,q,$s] = (policy_plan[i,l,m,x,y,z,q,$s-1] == 1 ? policy_window[i,l,m,x,y,z,$s-$ra+6+$init_t-1,2] : 0)
        @tullio v[i,l,m,x,y,z,q,$s] = (policy_plan[i,l,m,x,y,z,q,$s-1] == 1 ? v_window[i,l,m,x,y,z,$s-$ra+6+$init_t-1] : (policy_plan[i,l,m,x,y,z,q,$s-1] == 2 ? v_retire_no_pension[i,$s-$ra+6+$init_t-2] : v_retire_with_pension[i,y,m,q,$s-$ra+6+$init_t-2]))
    end

    for s in ra+5-init_t+1+1:T-init_t+1
        @tullio policy_asset[i,l,m,x,y,z,q,$s] = policy_plan[i,l,m,x,y,z,q,$s-1] == 2 ? policy_retire_no_pension[i,$s-$ra+$init_t-1-5] : policy_retire_with_pension[i,y,m,q,$s-$ra+$init_t-1-5]
        @tullio v[i,l,m,x,y,z,q,$s] = policy_plan[i,l,m,x,y,z,q,$s-1] == 2 ? v_retire_no_pension[i,$s-$ra+$init_t-1-5] : v_retire_with_pension[i,y,m,q,$s-$ra+$init_t-1-5]
    end
    solution = @with_kw (policy_asset = policy_asset, policy_work = policy_work, policy_plan = policy_plan, ϵ_grid = ϵ_grid, v = v)
    return solution()
end

function initial_distribution(;para, init_para)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra) = para
    (; μ_a, σ_a, μ_ϵ, σ_ϵ, p_work, p_wy, μ_aime, σ_aime, μ_lme, σ_lme) = init_para

    ξ, w = gausshermite(ξ_nodes)
    

    asset_dist = truncated(Normal(μ_a, σ_a), minimum(asset), maximum(asset)) 
    ϵ_dist = truncated(Normal(μ_ϵ, σ_ϵ), minimum(ϵ), maximum(ϵ))
    work_dist = Bernoulli(p_work) 
    wy_dist = Categorical(p_wy)
    aime_dist = truncated(Normal(μ_aime, σ_aime), minimum(aime), maximum(aime)) 
    lme_dist = truncated(Normal(μ_lme, σ_lme), minimum(lme), maximum(lme))
    ξ_dist = truncated(Normal(0, σ), sqrt(2)*σ*minimum(ξ), sqrt(2)*σ*maximum(ξ))

    dists = @with_kw (asset_dist = asset_dist, ϵ_dist = ϵ_dist, work_dist = work_dist, wy_dist = wy_dist, aime_dist = aime_dist, lme_dist = lme_dist, ξ_dist = ξ_dist) 
    return dists()
end

function simulate(;dists, solution, para, n)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra) = para
    (; asset_dist, ϵ_dist, work_dist, wy_dist, aime_dist, lme_dist, ξ_dist) = dists
    (; policy_asset, policy_work, policy_plan, ϵ_grid, v) = solution

    real_retire_age = collect(-5:5)

    #variables 
    asset_path = zeros(n, T-init_t+2)
    wage_path = zeros(n, T-init_t+1)
    work_path = zeros(n, T-init_t+2) 
    wy_path = zeros(n, T-init_t+2)
    aime_path = zeros(n, T-init_t+2)
    lme_path = zeros(n, T-init_t+2)
    ϵ_path = zeros(n, T-init_t+2)
    plan_path = ones(n, T-init_t+2)
    consumption_path = zeros(n, T-init_t+1)
    v_path = zeros(n, T-init_t+1)
    retire_age = zeros(Int64, n)

    #initial Distributions
    asset_path[:,1] = rand(asset_dist, n)
    ϵ_path[:,1] = rand(ϵ_dist, n) 
    work_path[:,1] = rand(work_dist, n)
    wy_path[:,1] = rand(wy_dist, n) .- 1
    aime_path[:,1] = rand(aime_dist, n)
    lme_path[:,1] = rand(lme_dist, n) 

    #wage shocks
    ξ = rand(ξ_dist, n, T-init_t+1)

    #simulate 
    for s in 2:T-init_t+2
        t = s + init_t - 2
        wy_comp = δ[1] + δ[2]*t + δ[3]*t^2
        wage_path[:,s-1] = max.(exp.(ϵ_path[:,s-1] .+ wy_comp), 2.747)
        asset_func = LinearInterpolation((asset, ϵ_grid[:,s-1], wy, work, aime, lme, real_retire_age), policy_asset[:,:,:,:,:,:,:,s-1])
        work_func = LinearInterpolation((asset, ϵ_grid[:,s-1], wy, work, aime, lme, real_retire_age), policy_work[:,:,:,:,:,:,:,s-1])
        plan_func = LinearInterpolation((asset, ϵ_grid[:,s-1], wy, work, aime, lme, real_retire_age), policy_plan[:,:,:,:,:,:,:,s-1])
        asset_path[:,s] = min.(asset_func.(asset_path[:,s-1], ϵ_path[:,s-1], wy_path[:,s-1], work_path[:,s-1], aime_path[:,s-1], lme_path[:,s-1], min(max(t-ra,-5),5)*ones(n)), asset[end])
        work_path[:,s] = round.(work_func.(asset_path[:,s-1], ϵ_path[:,s-1], wy_path[:,s-1], work_path[:,s-1], aime_path[:,s-1], lme_path[:,s-1], min(max(t-ra,-5),5)*ones(n)))
        plan_path[:,s] = round.(plan_func.(asset_path[:,s-1], ϵ_path[:,s-1], wy_path[:,s-1], work_path[:,s-1], aime_path[:,s-1], lme_path[:,s-1], min(max(t-ra,-5),5)*ones(n)))
        wy_path[:,s] = min.(wy_path[:,s-1] .+ work_path[:,s-1], wy_ceil)
        @tullio c_aime[i] := (wage_path[i,$s-1] < aime[2] ? aime[1] : (wage_path[i,$s-1] < aime[3] ? aime[2] : aime[3]))
        pension_tax = c_aime*0.2*τ 
        lme_path[:,s] = min.(4.58, max.(lme_path[:,s-1], c_aime .* work_path[:,s])) 
        @tullio less_5y_f_aime[i] := c_aime[i]*work_path[i,$s]/(wy_path[i,$s-1]+work_path[i,$s]) + wy_path[i,$s-1]/(wy_path[i,$s-1]+work_path[i,$s])*aime_path[i,$s-1]
        @tullio more_5y_f_aime[i] := aime_path[i,$s-1] + 0.2*max(0, c_aime[i]*work_path[i,$s]-lme_path[i,$s-1])
        @tullio aime_path[i,$s] = (((wy_path[i,$s-1] + work_path[i,$s]) > 0) ? (wy_path[i,$s-1] < 5 ? less_5y_f_aime[i] : more_5y_f_aime[i]) : 2.747)
        @tullio aime_path[i,$s] = min(4.58, aime_path[i,$s])
        @tullio retire_age[i] = (plan_path[i,$s] > plan_path[i,$s-1] ? $s+$init_t-2 : retire_age[i])
        @tullio monthly_benefit[i] := max(wy_path[i,$s-1]*aime_path[i,$s-1]*0.00775+3, wy_path[i,$s-1]*aime_path[i,$s-1]*0.0155)*(1+0.04*(retire_age[i]-$ra))
        @tullio lumpsum_benefit[i] := min(max(wy_path[i,$s-1], 2*wy_path[i,$s-1]-15), 50)*aime_path[i,$s-1]*(1+0.04*(retire_age[i]-$ra))
        @tullio adj_cost[i] := $φ_l*(work_path[i,$s]-work_path[i,$s-1] == 1)
        consumption_path[:,s-1] = (1+r)*asset_path[:,s-1] + wage_path[:,s-1].*work_path[:,s] + (plan_path[:,s] .== 2).*lumpsum_benefit + (wy_path[:,s-1] .≥ 15).*(plan_path[:,s] .== 3).*monthly_benefit - pension_tax.*work_path[:,s].*(plan_path[:,s] .== 1) - adj_cost - asset_path[:,s]
        ϵ_path[:,s] = ρ*ϵ_path[:,s-1] + ξ[:,s-1]
        v_func = LinearInterpolation((asset, ϵ_grid[:,s-1], wy, work, aime, lme, real_retire_age), v[:,:,:,:,:,:,:,s-1])
        v_path[:,s-1] = v_func.(asset_path[:,s-1], ϵ_path[:,s-1], wy_path[:,s-1], work_path[:,s-1], aime_path[:,s-1], lme_path[:,s-1], min(max(t-ra,-5),5)*ones(n))
    end
    path = @with_kw (asset_path = asset_path, wage_path = wage_path, work_path = work_path, wy_path = wy_path, aime_path = aime_path, lme_path = lme_path, ϵ_path = ϵ_path, plan_path = plan_path, consumption_path = consumption_path, retire_age = retire_age, v_path = v_path) 
    return path()
end

para = @with_kw (γ = 3.0, η = 0.99, r = 0.02, β = 0.98, ξ_nodes = 20, ϵ = range(0.0, 3.0, 5),
    T = T, μ = mort, init_t = 40, ρ = 0.97, σ = 0.06, asset = collect(range(0.0, 30.0, 21)), 
    work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.1, δ = [-1.0, 0.06, -0.0006], φ_l = 5.0, θ_b = 5.0, κ = 2.0,
    aime = profile, plan = collect(1:3), ra = 65, τ = 0.12, lme = profile, fra = 65)
para = para()

sol = solve(;para)
int_sol = integrate_sol(sol;para)
init_para = @with_kw (μ_a = 3.0, σ_a = 5.0, μ_ϵ = 0.1, σ_ϵ = 0.2, p_work = 0.7, p_wy = rand(11) |> (x -> x ./ sum(x)), μ_aime = 2.8, σ_aime = 0.5, μ_lme = 2.8, σ_lme = 0.1)
init_para = init_para()
dists = initial_distribution(;para, init_para)

n = 1000
path = simulate(dists = dists, solution = int_sol, para = para, n = n)
df = DataFrame(id = vec(transpose(repeat(collect(1:n),1,para.T-para.init_t+1))), age = repeat(collect(para.init_t:para.T),n), 
    asset = vec(transpose(path.asset_path[:,2:end])), wage = vec(transpose(path.wage_path)), 
    work = vec(transpose(path.work_path[:,2:end])), work_year = vec(transpose(path.wy_path[:,2:end])), 
    aime = vec(transpose(path.aime_path[:,2:end])), lme = vec(transpose(path.lme_path[:,2:end])), 
    plan = vec(transpose(path.plan_path[:,2:end])), consumption = vec(transpose(path.consumption_path)), 
    retire_age = vec(transpose(repeat(path.retire_age, 1, para.T-para.init_t+1))), value = vec(transpose(path.v_path)))

list = filter(row -> row.value < -1e36, df).id |> unique
dirty_df = filter(row -> row.id in list, df)
clean_df = filter(row -> !(row.id in list), df)


group = groupby(clean_df, :id)
retire_ages = combine(group, :retire_age => last => :retire_age, :work_year => last => :work_year)

k = 135
fig, ax = lines(group[k].age, group[k].asset, label = "Asset")
lines!(ax, group[k].age, group[k].wage, label = "Wage")
vspan!(ax, filter(row -> row.work == 0, group[k]).age .- 0.5, filter(row -> row.work == 0, group[k]).age .+ 0.5, color = (:gray, 0.3), label = "Unemployed")
vlines!(ax, group[k].retire_age[1], color = (group[k].plan[end] == 2 ? :purple : :green), label = (group[k].plan[end] == 2 ? "Pension Type 2" : "Pension Type 3")) 
lines!(ax, group[k].age, group[k].consumption, label = "Consumption")
fig[1,2] = Legend(fig, ax, framevisible = false)
fig
hist(filter(row -> row.work_year != 0, retire_ages).retire_age, bins = 10, 
    title = "Retirement Age", normalization = :pdf, bar_labels = :values)
