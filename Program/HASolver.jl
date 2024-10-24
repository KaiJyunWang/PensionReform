#-------------------------------------------
# This is the solver and simulation file of a heterogeneous agent model. 
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
using BenchmarkTools, Interpolations
using JLD2, LaTeXStrings

#life-cycle problem of pension solver
mort = mortality([1.13, 64200, 0.1, 0.0002])
T = life_ceil([1.13, 64200, 0.1, 0.0002])

# age distribution
function survival(t::Int) 
    if t ≤ 1 
        return 1 
    else 
        return prod(1 .- mort[1:t-1]) 
    end
end

function age_dist(t::Int, t_0::Int=25)
    return survival(t)/sum([survival(i) for i in t_0:T])
end


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
    (; γ, η, r, ρ, σ, β, ξ_nodes, ϵ, T, μ, init_t, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra, reduction) = para
    #initialization 
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
    policy_retire_with_pension = zeros(length(β), length(asset), length(aime), length(wy), length(real_retire_age), T-ra+5)
    policy_retire_no_pension = zeros(length(β), length(asset), T-ra+5)
    policy_window = zeros(length(β), length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(real_retire_age), 3)
    policy_before_window = zeros(length(β), length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(init_t:ra-6), 2)

    #value functions
    v_retire_with_pension = zeros(length(β), length(asset), length(aime), length(wy), length(real_retire_age), T-ra+6)
    v_retire_no_pension = zeros(length(β), length(asset), T-ra+6) 
    v_window = zeros(length(β), length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(real_retire_age))
    v_before_window = zeros(length(β), length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(init_t:ra-6))  

    #states
    type = collect(1:3)
    type_cua, β_cua, η_cua, asset_cua, work_cua, wy_cua, plan_cua, aime_cua, lme_cua = cu(type), cu(β), cu(η), cu(asset), cu(work), cu(wy), cu(plan), cu(aime), cu(lme)

    f_v_work = zeros(length(β), length(asset), length(work), length(ϵ), length(wy), length(aime), length(lme), ξ_nodes) |> x -> CuArray(x)
    #pre-computing
    #retired monthly benefit
    @tullio monthly_benefit[y,m,q] := 12*max(wy_cua[m]*aime[y]*0.00775+3, wy_cua[m]*aime[y]*0.0155)*(1+0.04*(real_retire_age[q]))*(1-$reduction)
    #lump-sum benefit
    @tullio lumpsum_benefit[y,m,q] := min(max(wy_cua[m], 2*wy_cua[m]-15), 50)*aime[y]*(1+0.04*(real_retire_age[q]))*(1-$reduction)
    #adjustment cost of working
    @tullio adj_cost[k,x] := $φ_l*(work_cua[k]-work_cua[x] == 1)
    #bequest
    @tullio bequest[b,j] := β_cua[b]*($θ_b*($κ+asset_cua[j])^(1-$γ))/(1-$γ)
    #future work years
    @tullio f_wy[m,k] := min(wy_cua[m]+work_cua[k], $wy_ceil)
    
    #VFI for retired with pension 
    for s in 1:T-ra+5
        t = T-s+1
        mort = μ[t]
        new_vr = v_retire_with_pension[:,:,:,:,:,end-s+1] |> x -> CuArray(x)
        
        @tullio retired_with_pension_consumption[i,j,y,m,q] := (1+$r)*asset_cua[i] + monthly_benefit[y,m,q] - asset_cua[j]
        @tullio retired_with_pension_utility[b,i,j,y,m,q] := retired_with_pension_consumption[i,j,y,m,q] ≥ $c_min ? retired_with_pension_consumption[i,j,y,m,q]^((1-$γ)*η_cua[b])/(1-$γ) : -1e38
        @tullio candidate_retire_with_pension[b,i,y,m,q,j] := retired_with_pension_utility[b,i,j,y,m,q] + (1-$mort)*β_cua[b]*new_vr[b,j,y,m,q] + bequest[b,j]*$mort

        can_retire_with_pension = Array(candidate_retire_with_pension)

        for state in CartesianIndices(v_retire_with_pension[:,:,:,:,:,T-ra+5-s+1])
            v_retire_with_pension[state, T-ra+5-s+1], ind = findmax(can_retire_with_pension[state,:])
            policy_retire_with_pension[state, T-ra+5-s+1] = asset[ind]
        end
    end

    #VFI for retired without pension
    for s in 1:T-ra+5
        t = T-s+1
        mort = μ[t]
        new_vn = v_retire_no_pension[:,:,end-s+1] |> x -> CuArray(x)
        
        @tullio retired_no_pension_consumption[i,j] := (1+$r)*asset_cua[i] - asset_cua[j]
        @tullio retired_no_pension_utility[b,i,j] := retired_no_pension_consumption[i,j] ≥ $c_min ? retired_no_pension_consumption[i,j]^((1-$γ)*η_cua[b])/(1-$γ) : -1e38
        @tullio candidate_retire_no_pension[b,i,j] := retired_no_pension_utility[b,i,j] + (1-$mort)*β_cua[b]*new_vn[b,j] + bequest[b,j]*$mort

        can_retire_no_pension = Array(candidate_retire_no_pension)

        for state in CartesianIndices(v_retire_no_pension[:,:,T-ra+5-s+1])
            v_retire_no_pension[state, T-ra+5-s+1], ind = findmax(can_retire_no_pension[state,:])
            policy_retire_no_pension[state, T-ra+5-s+1] = asset[ind]
        end
    end

    #VFI for window
    for s in 1:length(real_retire_age)
        t = ra+6-s 
        wy_comp = δ[1] + δ[2]*t + δ[3]*t^2
        mort = μ[t]
        if s != 1 
            v_func = LinearInterpolation((Float32.(type), Float32.(asset), Float32.(ϵ_grid[:,t-init_t+2]), Float32.(wy), Float32.(work), Float32.(aime), Float32.(lme)), v_window[:,:,:,:,:,:,:,end-s+2])
            cu_v_func = cu(v_func)
        end 
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        v_retire_no_pension_cua = CuArray(v_retire_no_pension[:,:,length(real_retire_age)-s+2])
        v_retire_with_pension_cua = CuArray(v_retire_with_pension[:,:,:,:,length(real_retire_age)-s+1,length(real_retire_age)-s+2]) 

        #future ϵ, avg. 
        @tullio f_ϵ[l,h] := $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ_cua[h]
        #minimum wage is 2.747
        @tullio wage[l] := max(2.747, exp(ϵ_cua[l] + $wy_comp))
        @tullio force_retire[l] := (exp(ϵ_cua[l] + $wy_comp) < 2.747)&&($t ≥ $fra)
        #current wage transformed into aime
        @tullio c_aime[l] := (wage[l] < aime[2] ? aime[1] : (wage[l] < aime[3] ? aime[2] : aime[3])) 
        @tullio pension_tax[l] := c_aime[l]*0.2*$τ*12
        @tullio window_consumption[i,j,l,m,k,x,q,y] := (1+$r)*asset_cua[i] + 12*wage[l]*work_cua[k] + (plan_cua[q] == 2)*lumpsum_benefit[y,m,12-$s] + (wy_cua[m]≥15)*(plan_cua[q]==3)*monthly_benefit[y,m,12-$s] - pension_tax[l]*work_cua[k]*(plan_cua[q] == 1) - adj_cost[k,x] - asset_cua[j]
        @tullio window_utility[b,i,j,l,m,k,x,q,y] := (window_consumption[i,j,l,m,k,x,q,y] ≥ $c_min ? (((window_consumption[i,j,l,m,k,x,q,y]^η_cua[b])*((1-260/364*work_cua[k])^(1-η_cua[b])))^(1-$γ))/(1-$γ) : -1e38)
        #future lowest monthly wage
        @tullio f_lme[z,l,k] := min(4.58, max(lme_cua[z], wage[l]*work_cua[k]))
        #future aime
        @tullio less_5y_f_aime[y,l,m,k] := c_aime[l]*work_cua[k]/(wy_cua[m]+work_cua[k]) + wy_cua[m]/(wy_cua[m]+work_cua[k])*aime_cua[y]
        @tullio more_5y_f_aime[y,z,l,k] := aime_cua[y] + 0.2*max(0, c_aime[l]*work_cua[k]-lme_cua[z])
        @tullio f_aime[y,z,l,m,k] := (((wy_cua[m] + work_cua[k]) > 0) ? (wy_cua[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,k]) : 2.747)
        @tullio f_aime[y,z,l,m,k] = min(4.58, f_aime[y,z,l,m,k]) 
        if s == 1
            #70 must be the last year of working
            @tullio f_utility[b,j,k,l,m,y,z,q,h] := (plan_cua[q] == 1 ? -1e38 : ((plan_cua[q] == 3)&&(wy_cua[m]≥15) ? v_retire_with_pension_cua[b,j,y,m] : v_retire_no_pension_cua[b,j])) (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), q in 1:length(plan), h in 1:ξ_nodes)
        else
            #for itp purpose
            @tullio f_type_s[b,j,k,l,m,y,z,h] := type_cua[b] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_asset_s[b,j,k,l,m,y,z,h] := asset_cua[j] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_ϵ_s[b,j,k,l,m,y,z,h] := f_ϵ[l,h] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_wy_s[b,j,k,l,m,y,z,h] := f_wy[m,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_work_s[b,j,k,l,m,y,z,h] := work_cua[k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_aime_s[b,j,k,l,m,y,z,h] := f_aime[y,z,l,m,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            @tullio f_lme_s[b,j,k,l,m,y,z,h] := f_lme[z,l,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
            f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = parent(f_type_s), parent(f_asset_s), parent(f_ϵ_s), parent(f_wy_s), parent(f_work_s), parent(f_aime_s), parent(f_lme_s) 
            f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = cu(f_type_s), cu(f_asset_s), cu(f_ϵ_s), cu(f_wy_s), cu(f_work_s), cu(f_aime_s), cu(f_lme_s)

            f_v_work = cu_v_func.(f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s)
            @tullio f_utility[b,j,k,l,m,y,z,q,h] := ((force_retire[l] == 0)&&(plan_cua[q] == 1) ? f_v_work[b,j,k,l,m,y,z,h] : ((plan_cua[q] == 3)&&(wy_cua[m]≥15) ? v_retire_with_pension_cua[b,j,y,m] : v_retire_no_pension_cua[b,j]))
        end
        #EV
        @tullio EV[b,j,k,l,m,y,z,q] := w_cua[h]*f_utility[b,j,k,l,m,y,z,q,h]

        @tullio candidate_window[b,i,l,m,x,y,z,j,k,q] := ((($t == $ra + 5)&&(plan_cua[q] == 1))||((force_retire[l] == 0)&&(work_cua[k] == 1)) ? -1e38 : window_utility[b,i,j,l,m,k,x,q,y] + (1-$mort)*β_cua[b]*EV[b,j,k,l,m,y,z,q] + $mort*bequest[b,j])

        can_window = Array(candidate_window)

        for state in CartesianIndices(v_window[:,:,:,:,:,:,:,length(real_retire_age)-s+1])
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
    v_func = LinearInterpolation((Float32.(type), Float32.(asset), Float32.(ϵ_grid[:,t-init_t+2]), Float32.(wy), Float32.(work), Float32.(aime), Float32.(lme)), v_window[:,:,:,:,:,:,:,1])
    cu_v_func = cu(v_func)
    ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
    #future ϵ, avg. 
    @tullio f_ϵ[l,h] := $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ_cua[h]
    #minimum wage is 2.747
    @tullio wage[l] := max(2.747, exp(ϵ_cua[l] + $wy_comp))
    #current wage transformed into aime
    @tullio c_aime[l] := (wage[l] < aime[2] ? aime[1] : (wage[l] < aime[3] ? aime[2] : aime[3]))
    @tullio pension_tax[l] := c_aime[l]*0.2*$τ*12
    @tullio before_window_consumption[i,j,l,k,x] := (1+$r)*asset_cua[i] + 12*wage[l]*work_cua[k] - pension_tax[l]*work_cua[k] - adj_cost[k,x] - asset_cua[j]
    @tullio before_window_utility[b,i,j,l,k,x] := before_window_consumption[i,j,l,k,x] ≥ $c_min ? (((before_window_consumption[i,j,l,k,x]^η_cua[b])*((1-260/364*work_cua[k])^(1-η_cua[b])))^(1-$γ))/(1-$γ) : -1e38
    #future lowest monthly wage
    @tullio f_lme[z,l,k] := min(4.58, max(lme_cua[z], wage[l]*work_cua[k]))
    #future aime
    @tullio less_5y_f_aime[y,l,m,k] := c_aime[l]*work_cua[k]/(wy_cua[m]+work_cua[k]) + wy_cua[m]/(wy_cua[m]+work_cua[k])*aime_cua[y]
    @tullio more_5y_f_aime[y,z,l,k] := aime_cua[y] + 0.2*max(0, c_aime[l]*work_cua[k]-lme_cua[z])
    @tullio f_aime[y,z,l,m,k] := (((wy_cua[m] + work_cua[k]) > 0) ? (wy_cua[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,k]) : 2.747)
    @tullio f_aime[y,z,l,m,k] = min(4.58, f_aime[y,z,l,m,k])

    @tullio f_type_s[b,j,k,l,m,y,z,h] := type_cua[b] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_asset_s[b,j,k,l,m,y,z,h] := asset_cua[j] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_ϵ_s[b,j,k,l,m,y,z,h] := f_ϵ[l,h] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_wy_s[b,j,k,l,m,y,z,h] := f_wy[m,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_work_s[b,j,k,l,m,y,z,h] := work_cua[k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_aime_s[b,j,k,l,m,y,z,h] := f_aime[y,z,l,m,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    @tullio f_lme_s[b,j,k,l,m,y,z,h] := f_lme[z,l,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
    f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = parent(f_type_s), parent(f_asset_s), parent(f_ϵ_s), parent(f_wy_s), parent(f_work_s), parent(f_aime_s), parent(f_lme_s)
    f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = cu(f_type_s), cu(f_asset_s), cu(f_ϵ_s), cu(f_wy_s), cu(f_work_s), cu(f_aime_s), cu(f_lme_s)

    f_utility_before_window = cu_v_func.(f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s)
    
    #EV
    @tullio EV_before_window[b,j,k,l,m,y,z] := w_cua[h]*f_utility_before_window[b,j,k,l,m,y,z,h]
    @tullio candidate_before_window[b,i,l,m,x,y,z,j,k] := before_window_utility[b,i,j,l,k,x] + (1-$mort)*β_cua[b]*EV_before_window[b,j,k,l,m,y,z] + $mort*bequest[b,j] 

    can_before_window = Array(candidate_before_window)

    for state in CartesianIndices(v_before_window[:,:,:,:,:,:,:,t-init_t+1])
        v_before_window[state, t-init_t+1], ind = findmax(can_before_window[state,:,:])
        policy_before_window[state, t-init_t+1,1] = asset[ind[1]]
        policy_before_window[state, t-init_t+1,2] = work[ind[2]]
    end

    for s in 2:ra-6-init_t+1
        t = ra-5-s
        mort = μ[t]
        wy_comp = δ[1] + δ[2]*t + δ[3]*t^2
        v_func = LinearInterpolation((Float32.(type), Float32.(asset), Float32.(ϵ_grid[:,t-init_t+2]), Float32.(wy), Float32.(work), Float32.(aime), Float32.(lme)), v_before_window[:,:,:,:,:,:,:,end-s+2])
        cu_v_func = cu(v_func)
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        #future ϵ, avg. 
        @tullio f_ϵ[l,h] := $ρ * ϵ_cua[l] + sqrt(2)* $σ * ξ_cua[h]
        #minimum wage is 2.747
        @tullio wage[l] := max(2.747, exp(ϵ_cua[l] + $wy_comp))
        #current wage transformed into aime
        @tullio c_aime[l] := (wage[l] < aime[2] ? aime[1] : (wage[l] < aime[3] ? aime[2] : aime[3]))
        @tullio pension_tax[l] := c_aime[l]*0.2*$τ*12
        @tullio before_window_consumption[i,j,l,k,x] := (1+$r)*asset_cua[i] + 12*wage[l]*work_cua[k] - pension_tax[l]*work_cua[k] - adj_cost[k,x] - asset_cua[j]
        @tullio before_window_utility[b,i,j,l,k,x] := before_window_consumption[i,j,l,k,x] ≥ $c_min ? (((before_window_consumption[i,j,l,k,x]^η_cua[b])*((1-260/364*work_cua[k])^(1-η_cua[b])))^(1-$γ))/(1-$γ) : -1e38
        #future lowest monthly wage
        @tullio f_lme[z,l,k] := min(4.58, max(lme_cua[z], wage[l]*work_cua[k]))
        #future aime
        @tullio less_5y_f_aime[y,l,m,k] := c_aime[l]*work_cua[k]/(wy_cua[m]+work_cua[k]) + wy_cua[m]/(wy_cua[m]+work_cua[k])*aime_cua[y]
        @tullio more_5y_f_aime[y,z,l,k] := aime_cua[y] + 0.2*max(0, c_aime[l]*work_cua[k]-lme[z])
        @tullio f_aime[y,z,l,m,k] := (((wy_cua[m] + work_cua[k]) > 0) ? (wy_cua[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,k]) : 2.747)
        @tullio f_aime[y,z,l,m,k] = min(4.58, f_aime[y,z,l,m,k])

        @tullio f_type_s[b,j,k,l,m,y,z,h] := type_cua[b] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_asset_s[b,j,k,l,m,y,z,h] := asset_cua[j] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_ϵ_s[b,j,k,l,m,y,z,h] := f_ϵ[l,h] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_wy_s[b,j,k,l,m,y,z,h] := f_wy[m,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_work_s[b,j,k,l,m,y,z,h] := work_cua[k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_aime_s[b,j,k,l,m,y,z,h] := f_aime[y,z,l,m,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        @tullio f_lme_s[b,j,k,l,m,y,z,h] := f_lme[z,l,k] (b in 1:length(β), j in 1:length(asset), k in 1:length(work), l in 1:length(ϵ), m in 1:length(wy), y in 1:length(aime), z in 1:length(lme), h in 1:ξ_nodes)
        f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = parent(f_type_s), parent(f_asset_s), parent(f_ϵ_s), parent(f_wy_s), parent(f_work_s), parent(f_aime_s), parent(f_lme_s)
        f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s = cu(f_type_s), cu(f_asset_s), cu(f_ϵ_s), cu(f_wy_s), cu(f_work_s), cu(f_aime_s), cu(f_lme_s)

        f_utility_before_window = cu_v_func.(f_type_s, f_asset_s, f_ϵ_s, f_wy_s, f_work_s, f_aime_s, f_lme_s)
        #EV
        @tullio EV_before_window[b,j,k,l,m,y,z] := w_cua[h]*f_utility_before_window[b,j,k,l,m,y,z,h]
        @tullio candidate_before_window[b,i,l,m,x,y,z,j,k] := before_window_utility[b,i,j,l,k,x] + (1-$mort)*β_cua[b]*EV_before_window[b,j,k,l,m,y,z] + $mort*bequest[b,j]

        can_before_window = Array(candidate_before_window)

        for state in CartesianIndices(v_before_window[:,:,:,:,:,:,:,t-init_t+1])
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
    (; γ, η, r, ρ, σ, β, ξ_nodes, ϵ, T, μ, init_t, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra, reduction) = para
    (; ϵ_grid, policy_before_window, policy_window, policy_retire_no_pension, policy_retire_with_pension, v_before_window, v_window, v_retire_no_pension, v_retire_with_pension) = sol
    real_retire_age = collect(-5:5)

    policy_plan = ones(length(β), length(asset), length(ϵ), length(wy), length(work), length(aime), length(lme), length(real_retire_age), T-init_t+1) 
    policy_asset = zeros(size(policy_plan))
    policy_work = zeros(size(policy_plan))
    v = zeros(size(policy_plan))
    policy_plan[:,:,:,:,:,:,:,:,ra-5-init_t+1:ra+5-init_t+1] = stack(fill(policy_window[:,:,:,:,:,:,:,:,3],11))
    
    for t in 2:T-init_t+1
        @tullio policy_plan[b,i,l,m,x,y,z,q,$t] = policy_plan[b,i,l,m,x,y,z,q,$t-1] != 1 ? policy_plan[b,i,l,m,x,y,z,q,$t-1] : policy_plan[b,i,l,m,x,y,z,q,$t]
    end
    
    for s in 1:ra-6-init_t+1
        policy_asset[:,:,:,:,:,:,:,:,s] = stack(fill(policy_before_window[:,:,:,:,:,:,:,s,1],11))
        policy_work[:,:,:,:,:,:,:,:,s] = stack(fill(policy_before_window[:,:,:,:,:,:,:,s,2],11))
        v[:,:,:,:,:,:,:,:,s] = stack(fill(v_before_window[:,:,:,:,:,:,:,s],11))
    end

    for s in ra-6-init_t+2:ra+5-init_t+1
        @tullio policy_asset[b,i,l,m,x,y,z,q,$s] = (policy_plan[b,i,l,m,x,y,z,q,$s-1] == 1 ? policy_window[b,i,l,m,x,y,z,$s-$ra+6+$init_t-1,1] : (policy_plan[b,i,l,m,x,y,z,q,$s-1] == 2 ? policy_retire_no_pension[b,i,$s-$ra+6+$init_t-2] : policy_retire_with_pension[b,i,y,m,q,$s-$ra+6+$init_t-2]))
        @tullio policy_work[b,i,l,m,x,y,z,q,$s] = (policy_plan[b,i,l,m,x,y,z,q,$s-1] == 1 ? policy_window[b,i,l,m,x,y,z,$s-$ra+6+$init_t-1,2] : 0)
        @tullio v[b,i,l,m,x,y,z,q,$s] = (policy_plan[b,i,l,m,x,y,z,q,$s-1] == 1 ? v_window[b,i,l,m,x,y,z,$s-$ra+6+$init_t-1] : (policy_plan[b,i,l,m,x,y,z,q,$s-1] == 2 ? v_retire_no_pension[b,i,$s-$ra+6+$init_t-2] : v_retire_with_pension[b,i,y,m,q,$s-$ra+6+$init_t-2]))
    end

    for s in ra+5-init_t+1+1:T-init_t+1
        @tullio policy_asset[b,i,l,m,x,y,z,q,$s] = policy_plan[b,i,l,m,x,y,z,q,$s-1] == 2 ? policy_retire_no_pension[b,i,$s-$ra+$init_t-1-5] : policy_retire_with_pension[b,i,y,m,q,$s-$ra+$init_t-1-5]
        @tullio v[b,i,l,m,x,y,z,q,$s] = policy_plan[b,i,l,m,x,y,z,q,$s-1] == 2 ? v_retire_no_pension[b,i,$s-$ra+$init_t-1-5] : v_retire_with_pension[b,i,y,m,q,$s-$ra+$init_t-1-5]
    end
    solution = @with_kw (policy_asset = policy_asset, policy_work = policy_work, policy_plan = policy_plan, ϵ_grid = ϵ_grid, v = v)
    return solution()
end

function initial_distribution(;para, init_para)
    #parameters
    (; γ, η, r, ρ, σ, β, ξ_nodes, ϵ, T, μ, init_t, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra, reduction) = para
    (; p_type, μ_a, σ_a, μ_ϵ, σ_ϵ, p_work, p_wy, μ_aime, σ_aime, μ_lme, σ_lme) = init_para

    ξ, w = gausshermite(ξ_nodes)
    
    type_dist = Categorical(p_type)
    asset_dist = truncated(Normal(μ_a, σ_a), minimum(asset), maximum(asset)) 
    ϵ_dist = truncated(Normal(μ_ϵ, σ_ϵ), minimum(ϵ), maximum(ϵ))
    work_dist = Bernoulli(p_work) 
    wy_dist = Categorical(p_wy)
    aime_dist = truncated(Normal(μ_aime, σ_aime), minimum(aime), maximum(aime)) 
    lme_dist = truncated(Normal(μ_lme, σ_lme), minimum(lme), maximum(lme))
    ξ_dist = truncated(Normal(0, σ), sqrt(2)*σ*minimum(ξ), sqrt(2)*σ*maximum(ξ))

    dists = @with_kw (type_dist = type_dist, asset_dist = asset_dist, ϵ_dist = ϵ_dist, work_dist = work_dist, wy_dist = wy_dist, aime_dist = aime_dist, lme_dist = lme_dist, ξ_dist = ξ_dist) 
    return dists()
end

function simulate(;dists, solution, para, n, rng = 9527)
    #parameters
    (; γ, η, r, ρ, σ, β, ξ_nodes, ϵ, T, μ, init_t, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra, reduction) = para
    (; type_dist, asset_dist, ϵ_dist, work_dist, wy_dist, aime_dist, lme_dist, ξ_dist) = dists
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
    Random.seed!(rng)
    type = rand(type_dist, n)
    n_type = ncategories(type_dist)
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
        wage_path[:,s-1] = max.(exp.(ϵ_path[:,s-1] .+ wy_comp), 2.747)# monthly
        asset_func = LinearInterpolation((collect(1:n_type), asset, ϵ_grid[:,s-1], wy, work, aime, lme, real_retire_age), policy_asset[:,:,:,:,:,:,:,:,s-1])
        work_func = LinearInterpolation((collect(1:n_type), asset, ϵ_grid[:,s-1], wy, work, aime, lme, real_retire_age), policy_work[:,:,:,:,:,:,:,:,s-1])
        plan_func = LinearInterpolation((collect(1:n_type), asset, ϵ_grid[:,s-1], wy, work, aime, lme, real_retire_age), policy_plan[:,:,:,:,:,:,:,:,s-1])
        asset_path[:,s] = min.(asset_func.(type, asset_path[:,s-1], ϵ_path[:,s-1], wy_path[:,s-1], work_path[:,s-1], aime_path[:,s-1], lme_path[:,s-1], min(max(t-ra,-5),5)*ones(n)), asset[end])
        work_path[:,s] = round.(work_func.(type, asset_path[:,s-1], ϵ_path[:,s-1], wy_path[:,s-1], work_path[:,s-1], aime_path[:,s-1], lme_path[:,s-1], min(max(t-ra,-5),5)*ones(n)))
        plan_path[:,s] = round.(plan_func.(type, asset_path[:,s-1], ϵ_path[:,s-1], wy_path[:,s-1], work_path[:,s-1], aime_path[:,s-1], lme_path[:,s-1], min(max(t-ra,-5),5)*ones(n)))
        wy_path[:,s] = min.(wy_path[:,s-1] .+ work_path[:,s-1], wy_ceil)
        @tullio c_aime[i] := (wage_path[i,$s-1] < aime[2] ? aime[1] : (wage_path[i,$s-1] < aime[3] ? aime[2] : aime[3]))
        pension_tax = c_aime*0.2*τ*12 # per year
        lme_path[:,s] = min.(4.58, max.(lme_path[:,s-1], c_aime .* work_path[:,s])) 
        @tullio less_5y_f_aime[i] := c_aime[i]*work_path[i,$s]/(wy_path[i,$s-1]+work_path[i,$s]) + wy_path[i,$s-1]/(wy_path[i,$s-1]+work_path[i,$s])*aime_path[i,$s-1]
        @tullio more_5y_f_aime[i] := aime_path[i,$s-1] + 0.2*max(0, c_aime[i]*work_path[i,$s]-lme_path[i,$s-1])
        @tullio aime_path[i,$s] = (((wy_path[i,$s-1] + work_path[i,$s]) > 0) ? (wy_path[i,$s-1] < 5 ? less_5y_f_aime[i] : more_5y_f_aime[i]) : 2.747)
        @tullio aime_path[i,$s] = min(4.58, aime_path[i,$s])
        @tullio retire_age[i] = (plan_path[i,$s] > plan_path[i,$s-1] ? min($s+$init_t-2, 70) : retire_age[i])
        @tullio monthly_benefit[i] := 12*max(wy_path[i,$s-1]*aime_path[i,$s-1]*0.00775+3, wy_path[i,$s-1]*aime_path[i,$s-1]*0.0155)*(1+0.04*(retire_age[i]-$ra))*(1 - $reduction)
        @tullio lumpsum_benefit[i] := min(max(wy_path[i,$s-1], 2*wy_path[i,$s-1]-15), 50)*aime_path[i,$s-1]*(1+0.04*(retire_age[i]-$ra))*(1 - $reduction)
        @tullio adj_cost[i] := $φ_l*(work_path[i,$s]-work_path[i,$s-1] == 1)
        consumption_path[:,s-1] = (1+r)*asset_path[:,s-1] + 12*wage_path[:,s-1].*work_path[:,s] + (plan_path[:,s-1] .== 1) .* (plan_path[:,s] .== 2).*lumpsum_benefit + (wy_path[:,s-1] .≥ 15).*(plan_path[:,s] .== 3).*monthly_benefit - pension_tax.*work_path[:,s].*(plan_path[:,s] .== 1) - adj_cost - asset_path[:,s]
        ϵ_path[:,s] = ρ*ϵ_path[:,s-1] + ξ[:,s-1]
        v_func = LinearInterpolation((collect(1:n_type), asset, ϵ_grid[:,s-1], wy, work, aime, lme, real_retire_age), v[:,:,:,:,:,:,:,:,s-1])
        v_path[:,s-1] = v_func.(type, asset_path[:,s-1], ϵ_path[:,s-1], wy_path[:,s-1], work_path[:,s-1], aime_path[:,s-1], lme_path[:,s-1], min(max(t-ra,-5),5)*ones(n))
    end
    path = @with_kw (asset_path = asset_path, wage_path = 12*wage_path, work_path = work_path, wy_path = wy_path, aime_path = aime_path, lme_path = lme_path, ϵ_path = ϵ_path, plan_path = plan_path, consumption_path = consumption_path, retire_age = retire_age, v_path = v_path, type = type) 
    return path()
end
#=
HAP = @with_kw (γ = 3.0, η = [0.412, 0.649, 0.967], r = 0.02, ρ = 0.97, σ = 0.2, β = [0.945, 0.859, 1.124], ξ_nodes = 20, 
    ϵ = range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), 5), T = T, μ = mort, init_t = 25, asset = collect(exp.(range(0.0, 7.0, 71)) .- 1), 
    work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.3, δ = [-2.3, 0.13, -0.001], φ_l = 20.0, θ_b = 200, κ = 700,
    aime = profile, plan = collect(1:3), ra = 65, τ = 0.12, lme = profile, fra = 130, reduction = 0.0)
HAP = HAP() 

solution = solve(para = HAP)
int_sol = integrate_sol(solution; para = HAP)
init_para = @with_kw (p_type = [0.267, 0.615, 1-0.267-0.615], μ_a = 3.0, σ_a = 1e-5, μ_ϵ = 0.0, σ_ϵ = HAP.σ/sqrt(1-HAP.ρ^2), p_work = 1.0, p_wy = [1], μ_aime = 2.747, σ_aime = 1e-5, μ_lme = 2.747, σ_lme = 1e-5)
init_para = init_para()
dists = initial_distribution(;para = HAP, init_para = init_para)

n = 5000
path = simulate(dists = dists, solution = int_sol, para = HAP, n = n)
df = DataFrame(id = vec(transpose(repeat(collect(1:n),1,HAP.T-HAP.init_t+1))), age = repeat(collect(HAP.init_t:HAP.T),n), 
    asset = vec(transpose(path.asset_path[:,2:end])), wage = vec(transpose(path.wage_path)), 
    work = vec(transpose(path.work_path[:,2:end])), work_year = vec(transpose(path.wy_path[:,2:end])), 
    aime = vec(transpose(path.aime_path[:,2:end])), lme = vec(transpose(path.lme_path[:,2:end])), 
    plan = vec(transpose(path.plan_path[:,2:end])), consumption = vec(transpose(path.consumption_path)), 
    retire_age = vec(transpose(repeat(path.retire_age, 1, HAP.T-HAP.init_t+1))), value = vec(transpose(path.v_path)), 
    type = vec(transpose(repeat(path.type, 1, HAP.T-HAP.init_t+1))))

list = filter(row -> row.consumption < 0, df).id |> unique
dirty_df = filter(row -> row.id in list, df)
clean_df = filter(row -> !(row.id in list), df)

group = groupby(clean_df, :id)
retire_ages = combine(group, :retire_age => last => :retire_age, :work_year => last => :work_year)

begin
    k = 7
    fig, ax = lines(group[k].age, group[k].asset, label = "Asset")
    lines!(ax, group[k].age, group[k].wage, label = "Wage")
    vspan!(ax, filter(row -> row.work == 0, group[k]).age .- 0.5, filter(row -> row.work == 0, group[k]).age .+ 0.5, color = (:gray, 0.3), label = "Unemployed")
    vlines!(ax, group[k].retire_age[1], color = (group[k].plan[end] == 2 ? :purple : :green), label = (group[k].plan[end] == 2 ? "Pension Type 2" : "Pension Type 3")) 
    lines!(ax, group[k].age, group[k].consumption, label = "Consumption", color = :red)
    fig[1,2] = Legend(fig, ax, framevisible = false)
    fig
end
hist(filter(row -> row.work_year != 0, retire_ages).retire_age,
    title = "Retirement Age", normalization = :pdf)

begin
    k = 4
    fig, ax = lines(group[k].age, group[k].wage, label = "Wage")
    fig
end
=#

# simulation 
# population setting
N = 1000
periods = -5:10
ages = zeros(Int, length(periods), N)
ages[1,:] = rand(Categorical(age_dist.(25:T)), N) .+ 24 # initial ages
survive = ones(Int, length(periods), N)

for t in 2:16
    ages[t,:] = ages[t-1,:] .+ 1 
    survive[t,:] = (rand(N) .≥ mort[ages[t-1,:]]) .* survive[t-1,:]
end

# benchmark 
HAP = @with_kw (γ = 3.0, η = [0.412, 0.649, 0.967], r = 0.02, ρ = 0.97, σ = 0.2, β = [0.945, 0.859, 1.124], ξ_nodes = 20, 
    ϵ = range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), 5), T = T, μ = mort, init_t = 25, asset = collect(exp.(range(0.0, 7.0, 71)) .- 1), 
    work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.3, δ = [-2.3, 0.13, -0.001], φ_l = 20.0, θ_b = 200, κ = 700,
    aime = profile, plan = collect(1:3), ra = 65, τ = 0.12, lme = profile, fra = 130, reduction = 0.0)
HAP = HAP() 
# jldsave("benchmark_sol.jld2", sol = int_sol)
benchmark_sol = load("benchmark_sol.jld2")

init_para = @with_kw (p_type = [0.267, 0.615, 1-0.267-0.615], μ_a = 3.0, σ_a = 1e-5, μ_ϵ = 0.0, σ_ϵ = HAP.σ/sqrt(1-HAP.ρ^2), p_work = 1.0, p_wy = [1], μ_aime = 2.747, σ_aime = 1e-5, μ_lme = 2.747, σ_lme = 1e-5)
init_para = init_para()
dists = initial_distribution(;para = HAP, init_para = init_para)
path = simulate(dists = dists, solution = benchmark_sol["sol"], para = HAP, n = 30000)
df = DataFrame(id = vec(transpose(repeat(collect(1:30000),1,HAP.T-HAP.init_t+1))), age = repeat(collect(HAP.init_t:HAP.T),30000), 
    asset = vec(transpose(path.asset_path[:,2:end])), wage = vec(transpose(path.wage_path)), 
    ϵ = vec(transpose(path.ϵ_path[:,2:end])),
    work = vec(transpose(path.work_path[:,2:end])), work_year = vec(transpose(path.wy_path[:,2:end])), 
    aime = vec(transpose(path.aime_path[:,2:end])), lme = vec(transpose(path.lme_path[:,2:end])), 
    plan = vec(transpose(path.plan_path[:,2:end])), consumption = vec(transpose(path.consumption_path)), 
    retire_age = vec(transpose(repeat(path.retire_age, 1, HAP.T-HAP.init_t+1))), value = vec(transpose(path.v_path)), 
    type = vec(transpose(repeat(path.type, 1, HAP.T-HAP.init_t+1))))

list = filter(row -> row.consumption < 0, df).id |> unique
dirty_df = filter(row -> row.id in list, df)
clean_df = filter(row -> !(row.id in list), df)
clean_df.id = 1 .+ div.(0:nrow(clean_df)-1, T - 25 + 1)


group = groupby(clean_df, :id)
retire_ages = combine(group, :retire_age => last => :retire_age, :work_year => last => :work_year)
#=
begin
    k = 7
    fig, ax = lines(group[k].age, group[k].asset, label = "Asset")
    lines!(ax, group[k].age, group[k].wage, label = "Wage")
    vspan!(ax, filter(row -> row.work == 0, group[k]).age .- 0.5, filter(row -> row.work == 0, group[k]).age .+ 0.5, color = (:gray, 0.3), label = "Unemployed")
    vlines!(ax, group[k].retire_age[1], color = (group[k].plan[end] == 2 ? :purple : :green), label = (group[k].plan[end] == 2 ? "Pension Type 2" : "Pension Type 3")) 
    lines!(ax, group[k].age, group[k].consumption, label = "Consumption", color = :red)
    fig[1,2] = Legend(fig, ax, framevisible = false)
    fig
end
hist(filter(row -> row.work_year != 0, retire_ages).retire_age,
    title = "Retirement Age", normalization = :pdf)
=#
# periods -5 : 10
benchmark_df = DataFrame(id = vec(transpose(repeat(1:N, 1, 16))), period = vec(repeat(-5:10, N)), 
    age = vec(ages), survive = vec(survive))
gd_benchmark = groupby(benchmark_df, :id)
for i in 1:N
    gd_benchmark[i][!, :asset] = group[i].asset[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :wage] = group[i].wage[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :work] = group[i].work[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :work_year] = group[i].work_year[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :aime] = group[i].aime[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :lme] = group[i].lme[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :plan] = group[i].plan[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :consumption] = group[i].consumption[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :retire_age] = group[i].retire_age[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :type] = group[i].type[gd_benchmark[i].age .- 24]
    gd_benchmark[i][!, :ϵ] = group[i].ϵ[gd_benchmark[i].age .- 24]
end

agg_benchmark = combine(gd_benchmark, :asset, :wage, :work, :work_year, :aime, :lme, 
    :plan, :consumption, :retire_age, :type => first => :type, :survive, :age, :period, :ϵ)
select_working(x, y) = y == 1 ? x : missing
select_working_v(x,y) = select_working.(x,y) 
avg_benchmark = combine(groupby(agg_benchmark, :period), :asset => mean => :asset,
    :work => mean => :work, [:wage, :work] => mean ∘ skipmissing ∘ select_working_v => :wage, :work_year => mean => :work_year, 
    :aime => mean => :aime, :lme => mean => :lme, :plan => mean => :plan, 
    :consumption => mean => :consumption, :retire_age => mean => :retire_age, 
    :survive => mean => :survive)

avg_benchmark.retire_age
# reducing benefits 
# reduction = 0.2
HAP_rb20 = @with_kw (γ = 3.0, η = [0.412, 0.649, 0.967], r = 0.02, ρ = 0.97, σ = 0.2, β = [0.945, 0.859, 1.124], ξ_nodes = 20, 
ϵ = range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), 5), T = T, μ = mort, init_t = 25, asset = collect(exp.(range(0.0, 7.0, 71)) .- 1), 
work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.3, δ = [-2.3, 0.13, -0.001], φ_l = 20.0, θ_b = 200, κ = 700,
aime = profile, plan = collect(1:3), ra = 65, τ = 0.12, lme = profile, fra = 130, reduction = 0.2)
HAP_rb20 = HAP_rb20()
rb20_solution = solve(para = HAP_rb20)
rb20_sol = integrate_sol(rb20_solution; para = HAP_rb20)
# jldsave("rb20.jld2", sol = rb20_sol)

rb20_sol = load("rb20.jld2")["sol"]
length(gd_benchmark)
# simulate one period ahead
function simulate_one_period(df; para, sol, new_ϵ, new_wage)
    #parameters
    (; γ, η, r, ρ, σ, β, ξ_nodes, ϵ, T, μ, init_t, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme, fra, reduction) = para
    (; policy_asset, policy_work, policy_plan, ϵ_grid, v) = sol

    n_type = 3
    new_asset = copy(df.asset)
    new_work = copy(df.work)
    new_plan = copy(df.plan)
    new_work_year = copy(df.work_year)
    new_aime = copy(df.aime)
    new_lme = copy(df.lme)
    new_consumption = copy(df.consumption)
    new_retire_age = copy(df.retire_age)

    real_retire_age = collect(-5:5)

    for i in 1:nrow(df)
        asset_func = LinearInterpolation((collect(1:n_type), asset, ϵ_grid[:,df.age[i]-24], wy, work, aime, lme, real_retire_age), policy_asset[:,:,:,:,:,:,:,:,df.age[i]-24])
        work_func = LinearInterpolation((collect(1:n_type), asset, ϵ_grid[:,df.age[i]-24], wy, work, aime, lme, real_retire_age), policy_work[:,:,:,:,:,:,:,:,df.age[i]-24])
        plan_func = LinearInterpolation((collect(1:n_type), asset, ϵ_grid[:,df.age[i]-24], wy, work, aime, lme, real_retire_age), policy_plan[:,:,:,:,:,:,:,:,df.age[i]-24])
        new_asset[i] = min(asset_func(df.type[i], df.asset[i], new_ϵ[i], df.work_year[i], df.work[i], df.aime[i], df.lme[i], clamp(df.age[i]-ra, -5, 5)), asset[end])
        new_work[i] = round(work_func(df.type[i], df.asset[i], new_ϵ[i], df.work_year[i], df.work[i], df.aime[i], df.lme[i], clamp(df.age[i]-ra, -5, 5)))
        new_plan[i] = df.plan[i] == 1 ? round(plan_func(df.type[i], df.asset[i], new_ϵ[i], df.work_year[i], df.work[i], df.aime[i], df.lme[i], clamp(df.age[i]-ra, -5, 5))) : df.plan[i]
        new_work_year[i] = min(df.work_year[i] + new_work[i], wy_ceil)
        c_aime = new_wage[i] < aime[2] ? aime[1] : (new_wage[i] < aime[3] ? aime[2] : aime[3])
        pension_tax = c_aime*0.2*τ*12

        new_lme[i] = min.(4.58, max.(df.lme[i], c_aime * df.work[i])) 
        less_5y_f_aime = c_aime*new_work[i]/(df.work_year[i] + new_work[i]) + df.work_year[i]/(df.work_year[i] + new_work[i])*df.aime[i]
        more_5y_f_aime = df.aime[i] + 0.2*max(0, c_aime * new_work[i]-df.lme[i])
        new_aime[i] = (((df.work_year[i] + new_work[i]) > 0) ? (df.work_year[i] < 5 ? less_5y_f_aime : more_5y_f_aime) : 2.747)
        new_aime[i] = min(4.58, new_aime[i])
        new_retire_age[i] = (new_plan[i] > df.plan[i] ? min(df.age[i]+1, 70) : df.retire_age[i])
        monthly_benefit = 12*max(df.work_year[i]*df.aime[i]*0.00775+3, df.work_year[i]*df.aime[i]*0.0155)*(1+0.04*(new_retire_age[i]-ra))*(1 - reduction)
        lumpsum_benefit = min(max(df.work_year[i], 2*df.work_year[i]-15), 50)*df.aime[i]*(1+0.04*(new_retire_age[i]-ra))*(1 - reduction)
        adj_cost = φ_l*(new_work[i]-df.work[i] == 1)
        new_consumption[i] = (1+r)*new_asset[i] + 12*new_wage[i]*new_work[i] + (df.age[i] - ra ≤ 5)*(df.plan[i] == 1) * (new_plan[i] == 2)*lumpsum_benefit + (df.age[i] - ra ≤ 5)*(df.work_year[i] ≥ 15)*(new_plan[i] == 3)*monthly_benefit - pension_tax * new_work[i] * (new_plan[i] == 1) - adj_cost - new_asset[i]
    end
    return (; asset = new_asset, work = new_work, plan = new_plan, work_year = new_work_year, aime = new_aime, lme = new_lme, consumption = new_consumption, retire_age = new_retire_age)
end

rb_20 = filter(row -> row.period == 0, agg_benchmark)
sim_20 = simulate_one_period(rb_20, para = HAP_rb20, sol = rb20_sol, new_ϵ = agg_benchmark.ϵ[agg_benchmark.period .== 1], new_wage = agg_benchmark.wage[agg_benchmark.period .== 1])

agg_rb_20 = agg_benchmark

gd_rb_20 = groupby(agg_rb_20, :period)
gd_rb_20[7][!, :asset] = sim_20.asset
groupby(agg_benchmark, :period)[7].asset == sim_20.asset
groupby(agg_benchmark, :period)[7].consumption 
sim_20.consumption
agg_rb_20
for t in 0:last(periods)-1
    new_df
end


# reduction = 0.4
HAP_rb40 = @with_kw (γ = 3.0, η = [0.412, 0.649, 0.967], r = 0.02, ρ = 0.97, σ = 0.2, β = [0.945, 0.859, 1.124], ξ_nodes = 20, 
ϵ = range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), 5), T = T, μ = mort, init_t = 25, asset = collect(exp.(range(0.0, 7.0, 71)) .- 1), 
work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.3, δ = [-2.3, 0.13, -0.001], φ_l = 20.0, θ_b = 200, κ = 700,
aime = profile, plan = collect(1:3), ra = 65, τ = 0.12, lme = profile, fra = 130, reduction = 0.4)
HAP_rb40 = HAP_rb40()
rb40_solution = solve(para = HAP_rb40)
rb40_sol = integrate_sol(rb40_solution; para = HAP_rb40)
jldsave("rb40.jld2", sol = rb40_sol)


# reduction = 0.6
HAP_rb60 = @with_kw (γ = 3.0, η = [0.412, 0.649, 0.967], r = 0.02, ρ = 0.97, σ = 0.2, β = [0.945, 0.859, 1.124], ξ_nodes = 20, 
ϵ = range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), 5), T = T, μ = mort, init_t = 25, asset = collect(exp.(range(0.0, 7.0, 71)) .- 1), 
work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.3, δ = [-2.3, 0.13, -0.001], φ_l = 20.0, θ_b = 200, κ = 700,
aime = profile, plan = collect(1:3), ra = 65, τ = 0.12, lme = profile, fra = 130, reduction = 0.6)
HAP_rb60 = HAP_rb60()
rb60_solution = solve(para = HAP_rb60)
rb60_sol = integrate_sol(rb60_solution; para = HAP_rb60)
jldsave("rb60.jld2", sol = rb60_sol)


# defer retirement
# ra = 68
HAP_dr3 = @with_kw (γ = 3.0, η = [0.412, 0.649, 0.967], r = 0.02, ρ = 0.97, σ = 0.2, β = [0.945, 0.859, 1.124], ξ_nodes = 20, 
ϵ = range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), 5), T = T, μ = mort, init_t = 25, asset = collect(exp.(range(0.0, 7.0, 71)) .- 1), 
work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.3, δ = [-2.3, 0.13, -0.001], φ_l = 20.0, θ_b = 200, κ = 700,
aime = profile, plan = collect(1:3), ra = 68, τ = 0.12, lme = profile, fra = 130, reduction = 0.0)
HAP_dr3 = HAP_dr3()
dr3_solution = solve(para = HAP_dr3)
dr3_sol = integrate_sol(dr3_solution; para = HAP_dr3)
jldsave("dr3.jld2", sol = dr3_sol)

# ra = 70
HAP_dr5 = @with_kw (γ = 3.0, η = [0.412, 0.649, 0.967], r = 0.02, ρ = 0.97, σ = 0.2, β = [0.945, 0.859, 1.124], ξ_nodes = 20, 
ϵ = range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), 5), T = T, μ = mort, init_t = 25, asset = collect(exp.(range(0.0, 7.0, 71)) .- 1), 
work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.3, δ = [-2.3, 0.13, -0.001], φ_l = 20.0, θ_b = 200, κ = 700,
aime = profile, plan = collect(1:3), ra = 70, τ = 0.12, lme = profile, fra = 130, reduction = 0.0)
HAP_dr5 = HAP_dr5()
dr5_solution = solve(para = HAP_dr5)
dr5_sol = integrate_sol(dr5_solution; para = HAP_dr5)
jldsave("dr5.jld2", sol = dr5_sol)

# ra = 72
HAP_dr7 = @with_kw (γ = 3.0, η = [0.412, 0.649, 0.967], r = 0.02, ρ = 0.97, σ = 0.2, β = [0.945, 0.859, 1.124], ξ_nodes = 20, 
ϵ = range(-2*σ/sqrt(1-ρ^2), 2*σ/sqrt(1-ρ^2), 5), T = T, μ = mort, init_t = 25, asset = collect(exp.(range(0.0, 7.0, 71)) .- 1), 
work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.3, δ = [-2.3, 0.13, -0.001], φ_l = 20.0, θ_b = 200, κ = 700,
aime = profile, plan = collect(1:3), ra = 72, τ = 0.12, lme = profile, fra = 130, reduction = 0.0)
HAP_dr7 = HAP_dr7()
dr7_solution = solve(para = HAP_dr7)
dr7_sol = integrate_sol(dr7_solution; para = HAP_dr7)
jldsave("dr7.jld2", sol = dr7_sol)

