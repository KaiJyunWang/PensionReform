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
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, lme, scheme0, scheme1) = para 
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
    β_cua, η_cua, asset_cua, work_cua, wy_cua, plan_cua, aime_cua, lme_cua = cu(β), cu(η), cu(asset), cu(work), cu(wy), cu(plan), cu(aime), cu(lme)
    

end