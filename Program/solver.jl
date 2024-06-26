include("Mortality.jl")
using .Mortality
include("PensionBenefit.jl")
using .PensionBenefit

using Distributions, LinearAlgebra, Plots, FastGaussQuadrature
using Parameters, Random, Tables, Profile
using CUDA, CUDAKernels, KernelAbstractions, Tullio, CUDA.Adapt
using BenchmarkTools, Interpolations, ProgressBars
using JLD2

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

para = @with_kw (γ = 3.0, η = 0.6, r = 0.03, β = 0.97, ξ_nodes = 20, ϵ = range(0.0, 3.0, 5),
    T = T, μ = mort, init_t = 40, ρ = 0.97, σ = 0.04, asset = collect(range(0.0, 25.0, 15)), 
    work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.1, δ = [0.7, 0.02, -0.0003], φ_l = 20.0, θ_b = 0.6, κ = 2.0,
    aime = profile, plan = collect(1:3), ra = 65, τ = 0.12, lme = profile, fra = 65)
para = para()

function solve(;para)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra) = para
    nan = false
    error = 0
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
    wy_retire_over_15 = cu(collect(15:30))

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
        @tullio retired_with_pension_utility[i,j,y,m,q] := retired_with_pension_consumption[i,j,y,m,q] ≥ $c_min ? retired_with_pension_consumption[i,j,y,m,q]^(1-$γ)/(1-$γ) : -1e38
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
        @tullio retired_no_pension_utility[i,j] := retired_no_pension_consumption[i,j] ≥ $c_min ? retired_no_pension_consumption[i,j]^(1-$γ)/(1-$γ) : -1e38
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
            
            CUDA.@sync f_v_work = cu_v_func.(f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s)
            if !prod(.!isnan.(f_v_work))
                nan = true
                error = f_v_work
                break
            end
            @tullio f_utility[j,k,l,m,y,z,q,h] := ((force_retire[l] == 0)&&(plan_cua[q] == 1) ? f_v_work[j,k,l,m,y,z,h] : ((plan_cua[q] == 3)&&(wy_cua[m]≥15) ? v_retire_with_pension_cua[j,y,m] : v_retire_no_pension_cua[j]))
        end
        #EV
        @tullio EV[j,k,l,m,y,z,q] := w_cua[h]*f_utility[j,k,l,m,y,z,q,h]

        @tullio candidate_window[i,l,m,x,y,z,j,k,q] := ((force_retire[l] == 0)&&(work_cua[k] == 1) ? -1e38 : window_utility[i,j,l,m,k,x,q,y] + (1-$mort)*$β*EV[j,k,l,m,y,z,q] + $mort*bequest[j])

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

    for state in CartesianIndices(v_before_window[:,:,:,:,:,:,end])
        v_window[state, end], ind = findmax(can_before_window[state,:,:])
        policy_before_window[state, end,1] = asset[ind[1]]
        policy_before_window[state, end,2] = work[ind[2]]
    end
    
    for s in 2:length(init_t:ra-6)
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
    
    solution = @with_kw (srwp = policy_retire_with_pension, srnp = policy_retire_no_pension, sw = policy_window, sbw = policy_before_window, vrwp = v_retire_with_pension, vrnp = v_retire_no_pension, vw = v_window, vbw = v_before_window, ϵ_grid = ϵ_grid, nan = nan, error = error)
    return solution()
end 

sol = solve(;para)


#To Do
function simulate(solution; para, n, seed, )
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra) = para
    (; srwp, srnp, sw, sbw, vrwp, vrnp, vw, vbw, ϵ_grid) = solution

end
#=
function solve_retired_with_pension(;para)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra) = para
    #settings
    #all possible pension benefits in monthly
    rra = collect(-5:5) |> x -> CuArray(x)
    wy_rwp = collect(15:30) |> x -> CuArray(x)
    monthly_benefit = zeros(length(aime), length(wy_rwp), length(-5:5)) |> x -> CuArray(x)
    @tullio monthly_benefit[y,m,q] = max(wy_rwp[m]*aime[y]*0.00775+3, wy_rwp[m]*aime[y]*0.0155)*(1+0.04*(rra[q]))
    asset_cua = asset |> x -> CuArray(x)
    consumption = zeros(length(asset), length(asset), length(aime), length(wy_rwp), length(-5:5)) |> x -> CuArray(x)
    utility = zeros(size(consumption)) |> x -> CuArray(x)
    bequest = zeros(length(asset)) |> x -> CuArray(x)
    candidate = zeros(length(asset), length(aime), length(wy_rwp), length(-5:5), length(asset)) |> x -> CuArray(x)

    vr = zeros(length(asset), length(aime), length(wy_rwp), length(-5:5), T-ra+7) 
    policy = zeros(length(asset), length(aime), length(wy_rwp), length(-5:5), T-ra+6) 

    for s in tqdm(1:T-ra+6)
        t = T-s+1
        mort = μ[t]
        new_vr = vr[:,:,:,:,T-ra+6-s+2] |> x -> CuArray(x)
        
        @tullio consumption[i,j,y,m,q] = (1+$r)*asset_cua[i] + monthly_benefit[y,m,q] - asset_cua[j]
        
        @tullio utility[i,j,y,m,q] = consumption[i,j,y,m,q] ≥ $c_min ? consumption[i,j,y,m,q]^(1-$γ)/(1-$γ) : -1e38
        @tullio bequest[j] = ($θ_b*($κ+asset_cua[j])^(1-$γ))/(1-$γ)
        @tullio candidate[i,y,m,q,j] = utility[i,j,y,m,q] + (1-$mort)*$β*new_vr[j,y,m,q] + $β*bequest[j]*$mort

        can = Array(candidate)

        for state in CartesianIndices(vr[:,:,:,:,T-ra+6-s+1])
            vr[state, T-ra+6-s+1], ind = findmax(can[state,:])
            policy[state, T-ra+6-s+1] = asset[ind]
        end
    end
    return (policy = policy, vr = vr)
end

srwp, vrwp = solve_retired_with_pension(;para)

#solve retire without pension
function solve_retired_no_pension(;para)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra) = para
    #settings
    #all possible pension benefits in monthly
    asset_cua = asset |> x -> CuArray(x)
    consumption = zeros(length(asset), length(asset)) |> x -> CuArray(x)
    utility = zeros(size(consumption)) |> x -> CuArray(x)
    bequest = zeros(length(asset)) |> x -> CuArray(x)
    candidate = zeros(length(asset), length(asset)) |> x -> CuArray(x)

    vr = zeros(length(asset), T-ra+7) 
    policy = zeros(length(asset), T-ra+6) 

    for s in tqdm(1:T-ra+6)
        t = T-s+1
        mort = μ[t]
        new_vr = vr[:,T-ra+6-s+2] |> x -> CuArray(x)
        
        @tullio consumption[i,j] = (1+$r)*asset_cua[i] - asset_cua[j]
        @tullio utility[i,j] = consumption[i,j] ≥ $c_min ? consumption[i,j]^(1-$γ)/(1-$γ) : -1e38
        @tullio bequest[j] = ($θ_b*($κ+asset_cua[j])^(1-$γ))/(1-$γ)
        @tullio candidate[i,j] = utility[i,j] + (1-$mort)*$β*new_vr[j] + $β*bequest[j]*$mort

        can = Array(candidate)

        for state in CartesianIndices(vr[:,T-ra+6-s+1])
            vr[state, T-ra+6-s+1], ind = findmax(can[state,:])
            policy[state, T-ra+6-s+1] = asset[ind]
        end
    end
    return (policy = policy, vr = vr)
end

srnp, vrnp = solve_retired_no_pension(;para)

function solve_pension_window(vrwp, vrnp, ϵ_grid;para)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra, benefit) = para
    #settings
    window = collect(-5:5) |> x -> CuArray(x)
    ξ, w = gausshermite(ξ_nodes)
    ξ = ξ |> x -> CuArray(x)
    w = w |> x -> CuArray(x)
    ϵ_grid = zeros(length(ϵ), length(window))
    ϵ_grid[:,1] = ϵ
    for t in 2:length(window)
        ϵ_grid[:,t] = range(ρ*ϵ_grid[1,t-1] + sqrt(2)*σ*minimum(ξ), ρ*ϵ_grid[end,t-1] + sqrt(2)*σ*maximum(ξ), length(ϵ))
    end

    #policy function 1: asset, 2: work, 3: plan
    policy_window = zeros(length(asset), length(ϵ), length(wy), length(plan), length(work), length(aime), length(lme), length(window), 3)
    v = zeros(length(asset), length(ϵ), length(wy), length(plan), length(work), length(aime), length(lme), length(window))
    #computing matrices
    lumpsum_benefit = zeros(length(aime), length(wy), length(plan))
    monthly_benefit = zeros(length(aime), length(wy), length(plan))
    benefit = zeros(length(aime), length(wy), length(plan))
    extra_benefit = zeros(length(plan)) |> x -> CuArray(x)
    adj_cost = zeros(length(work), length(work))
    f_ϵ = zeros(length(ϵ), ξ_nodes)
    wage = zeros(length(ϵ))
    c_aime = zeros(length(ϵ))
    pension_tax = zeros(length(ϵ))
    income = zeros(length(ϵ), length(wy), length(work), length(plan), length(aime))
    consumption = zeros(length(asset), length(asset), length(ϵ), length(wy), length(work), length(work), length(plan), length(aime))
    leisure = zeros(length(work))
    utility = zeros(length(asset), length(asset), length(work), length(ϵ), length(wy), length(work), length(plan), length(aime))
    bequest = zeros(length(asset))
    forbid = zeros(length(plan), length(plan), length(wy))
    f_lme = zeros(length(lme), length(ϵ), length(work))
    f_wy = zeros(length(wy), length(work))
    less_5y_f_aime = zeros(length(aime), length(ϵ), length(wy), length(work))
    more_5y_f_aime = zeros(length(aime), length(lme), length(ϵ), length(work))
    f_aime = zeros(length(aime), length(lme), length(ϵ), length(wy), length(work))
    f_utility = zeros(length(asset), length(work), length(ϵ), length(wy), length(aime), length(lme), length(plan), ξ_nodes)
    EV = zeros(length(asset), length(work), length(ϵ), length(wy), length(aime), length(lme), length(plan))
    candidate = zeros(length(asset), length(ϵ), length(wy), length(plan), length(work), length(aime), length(lme), length(asset), length(work), length(plan))

    #benefit schemes
    #2: lump-sum pension, 3: monthly pension
    #pinned down the benefit
    @tullio lumpsum_benefit[y,m,q] = (plan[q] == 2)*min(max(wy[m], 2*wy[m]-15), 50)*aime[y]
    @tullio monthly_benefit[y,m,q] = (plan[q] == 3)*max(wy[m]*aime[y]*0.00775+3, wy[m]*aime[y]*0.0155)
    #adjustment cost of working
    @tullio adj_cost[k,x] = $φ_l*(work[k]-work[x] == 1)
    #extra benefit
    function ex_benefit(t::Int64, ra::Int64, type::Int64)
        if type == 2
            if abs(t-ra) ≤ 5
                return 0.04*(t-ra)
            else
                return -1.0
            end
        elseif type == 3
            if abs(t-ra) ≤ 5
                return 0.04*(t-ra)
            else
                return -1.0
            end
        else 
            return -1.0
        end
    end

    
    for s in tqdm(1:length(window))
        t = ra+6-s
        wy_comp = δ[1] + δ[2]*t + δ[3]*t^2
        #extra_benefit should be pinned down    
        @tullio extra_benefit[q] = ex_benefit($t, $ra, plan[q])
        mort = μ[t]
        v_func = LinearInterpolation((asset, ϵ_grid[:,end], wy, plan, work, aime, lme), v[:,:,:,:,:,:,:,end-s+1])
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        vrnp_cua = CuArray(vrnp[:,ra+5+s])
        vrwp_cua = CuArray(vrwp[:,:,:,:,ra+5+s])
        
        #future ϵ, avg. 
        @tullio f_ϵ[l,h] = $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ[h]
        #minimum wage is 2.747
        @tullio wage[l] = max(2.747, exp(ϵ_cua[l] + $wy_comp))
        #current wage transformed into aime
        @tullio c_aime[l] = stair(wage[l]; c = aime)
        @tullio pension_tax[l] = c_aime[l]*0.2*$τ
        #benefit
        @tullio benefit[y,m,q] = lumpsum_benefit[y,m,q] + monthly_benefit[y,m,q]*(1+extra_benefit[q])
        #net income
        @tullio income[l,m,k,q,y] = wage[l]*work[k] + benefit[y,m,q] - pension_tax[l]*work[k]*(plan[q] == 1)
        #consumption
        @tullio consumption[i,j,l,m,k,x,q,y] = (1+$r)*asset[i] + income[l,m,k,q,y] - adj_cost[k,x] - asset[j]
        #leisure
        @tullio leisure[k] = (1-260/364*work[k])^(1-$η)
        #consumption floor
        @tullio utility[i,j,k,l,m,x,q,y] = consumption[i,j,l,m,k,x,q,y] ≥ $c_min ? (((consumption[i,j,l,m,k,x,q,y]^($η))*leisure[k])^(1-$γ))/(1-$γ) : -1e38
        #bequest
        @tullio bequest[j] = ($θ_b*($κ+asset[j])^(1-$γ))/(1-$γ)*$mort
        #forbidden path of applying for pension
        #1: unreceived, 2: lump-sum, 3: monthly, 4: received lump-sum
        @tullio forbid[p,q,m] = ((($t - $ra < -5)&&(plan[q] != 1))||((wy[m] == 0)&&((plan[q] != 1)||(plan[q] != 4)))||((plan[p] == 2)&&(plan[q] != 4))||((plan[p] == 3)&&(plan[q] != 3))||((plan[p] == 4)&&(plan[q] != 4)) ? -1e38 : 0.0)
        #future lowest monthly wage
        @tullio f_lme[z,l,k] = min(4.58, max(lme[z], wage[l]*work[k]))
        #future work years
        @tullio f_wy[m,k] = min(wy[m]+work[k], $wy_ceil)
        #future aime
        @tullio less_5y_f_aime[y,l,m,k] = c_aime[l]*work[k]/(wy[m]+work[k]) + wy[m]/(wy[m]+work[k])*aime[y]
        @tullio more_5y_f_aime[y,z,l,k] = aime[y] + 0.2*max(0, c_aime[l]*work[k]-lme[z])
        @tullio f_aime[y,z,l,m,k] = (((wy[m] + work[k]) > 0) ? (wy[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,k]) : 2.747)
        @tullio f_aime[y,z,l,m,k] = min(4.58, f_aime[y,z,l,m,k])
        #EV part
        if s == 1
            #70 must be the last year of working
            @tullio f_utility[j,k,l,m,y,z,q,h] = (n[k] == 1 ? -1e38 : (q == 3 ? vrwp[j,y,m] : vrnp[j]))
        else
            @tullio f_utility[j,k,l,m,y,z,q,h] = (n[k] == 1 ? v_func(asset[j], f_ϵ[l,h], f_wy[m,k], plan[q], work[k], f_aime[y,z,l,m,k], f_lme[z,l,k]) : (q == 3 ? vrwp[j,y,m] : vrnp[j]))
        end
        #Gauss Quadrature
        @tullio EV[j,k,l,m,y,z,q] = w[h]*f_utility[j,k,l,m,y,z,q,h]


        @tullio candidate[i,l,m,p,x,y,z,j,k,q] = utility[i,j,k,l,m,x,q,y] + (1-$mort)*$β*EV[j,k,l,m,y,z,q]/sqrt(π) + bequest[j] + forbid[p,q,m]
        
        #transform back to CPU
        can = Array(candidate)
        
        for state in CartesianIndices(v[:,:,:,:,:,:,:,t-init_t+1])
            ind = argmax(can[state,:,:,:])
            policy[state,t-init_t+1,1] = asset[ind[1]]
            policy[state,t-init_t+1,2] = work[ind[2]]
            policy[state,t-init_t+1,3] = plan[ind[3]]
            v[state, t-init_t+1] = can[state,ind]
        end
    end
    return (policy = policy, ϵ_grid = ϵ_grid, v = v)
end

#no plan
function solve_before_window(;para)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra, benefit) = para
    
end

sol = solve_pension_window(v; para)

#simulation
ϵ = zeros(para.T-para.init_t+1)
#initial ϵ
rng = Xoshiro(1234)
ϵ[1] = 0.3
for t in 2:para.T-para.init_t+1
    ϵ[t] = para.ρ*ϵ[t-1] + para.σ*randn(rng)
end
asset = zeros(para.T-para.init_t+2)
#initial asset
asset[1] = 10.0
work = zeros(para.T-para.init_t+2)
#initial working status
work[1] = 1
wy = zeros(para.T-para.init_t+2)
#initial work years
wy[1] = 12
plan = zeros(para.T-para.init_t+2)
#initial pension status
plan[1] = 1
s_aime = zeros(para.T-para.init_t+2)
#initial aime
s_aime[1] = 4.0
lme = zeros(para.T-para.init_t+2)
#initial lme
lme[1] = 3.5

#value check
value = zeros(para.T-para.init_t+1)
#wage
wage = zeros(para.T-para.init_t+1)
consumption = zeros(para.T-para.init_t+1)

for t in 1:para.T-para.init_t+1
    value[t] = LinearInterpolation((para.asset, sol.ϵ_grid[:,t], para.wy, para.plan, para.work, para.aime, para.lme), sol.v[:,:,:,:,:,:,:,t])(asset[t], ϵ[t], wy[t], plan[t], work[t], s_aime[t], lme[t])
    a_func = LinearInterpolation((para.asset, sol.ϵ_grid[:,t], para.wy, para.plan, para.work, para.aime, para.lme), sol.policy[:,:,:,:,:,:,:,t,1])
    w_func = LinearInterpolation((para.asset, sol.ϵ_grid[:,t], para.wy, para.plan, para.work, para.aime, para.lme), sol.policy[:,:,:,:,:,:,:,t,2])
    p_func = LinearInterpolation((para.asset, sol.ϵ_grid[:,t], para.wy, para.plan, para.work, para.aime, para.lme), sol.policy[:,:,:,:,:,:,:,t,3])
    asset[t+1] = a_func(asset[t], ϵ[t], wy[t], plan[t], work[t], s_aime[t], lme[t])
    work[t+1] = round(Int, w_func(asset[t], ϵ[t], wy[t], plan[t], work[t], s_aime[t], lme[t]))
    plan[t+1] = round(Int, p_func(asset[t], ϵ[t], wy[t], plan[t], work[t], s_aime[t], lme[t]))
    #state transition
    wy_comp = para.δ[1] + para.δ[2]*(t+39) + para.δ[3]*(t+39)^2
    wage[t] = max(exp(ϵ[t] + wy_comp), 2.747)
    c_aime = stair(wage[t]; c = para.aime)
    pension_tax = c_aime*0.2*para.τ
    wy[t+1] = min(para.wy_ceil, wy[t]+work[t+1])
    s_aime[t+1] = wy[t+1]+work[t+1] > 0 ? (wy[t] < 5 ? c_aime*work[t+1]/(wy[t]+work[t+1]) + wy[t]/(wy[t]+work[t+1])*s_aime[t] : s_aime[t] + 0.2*max(0, c_aime*work[t+1]-lme[t])) : 2.747
    s_aime[t+1] = min(4.58, s_aime[t+1])
    lme[t+1] = min(4.58, max(lme[t], wage[t]*work[t+1]))
    consumption[t] = (1+para.r)*asset[t] + wage[t]*work[t+1] - asset[t+1] + (1-work[t+1])*c_aime*0.2*para.τ*(plan[t+1] == 1) - para.φ_l*(work[t+1]-work[t] == 1)   
end

#plotting
age = para.init_t:para.T
plt = plot(age, wage, label = "wage", xlabel = "age", legend = :outertopright)
plot!(age, asset[2:end], label = "asset")
vcat(findall(x -> x == 0, work[2:end]) .+ para.init_t .- 1.5, findall(x -> x == 0, work[2:end]) .+ para.init_t .- 0.5) |> x -> sort(x) |> x -> vspan!(plt, x, color = :gray, label = "not working", alpha = 0.3)
first_pension = findfirst(x -> x != 1, plan) + para.init_t - 2 
if plan[first_pension - para.init_t + 1] == 2
    vline!(plt, [first_pension + 0.5], label = "Lump-sum", color = :purple)
else
    vline!(plt, [first_pension + 0.5], label = "Monthly", color = :green)
end
plot!(age, consumption, label = "consumption")
savefig(plt, "life-cycle.png")
findall(x -> x < para.c_min, consumption)
findall(x -> x == 0, work)
plot(age, value, label = "value")
=#