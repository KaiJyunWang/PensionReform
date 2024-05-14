include("Mortality.jl")
using .Mortality
include("PensionBenefit.jl")
using .PensionBenefit

using Distributions, LinearAlgebra, Plots, FastGaussQuadrature
using Parameters, Random, Tables, Profile
using CUDA, CUDAKernels, KernelAbstractions, Tullio
using BenchmarkTools, Interpolations, ProgressBars
using JLD2

#life-cycle problem of pension solver
mort = mortality([1.5, 1e8, 0.2, 0.0003])
T = life_ceil([1.5, 1e8, 0.2, 0.0003])

#profile of aime
profile = [2.747, 3.48, 4.58]

function stair(x::Float64; c::Array{Float64,1} = c)
    if x < c[2]
        return c[1]
    elseif x < c[3]
        return c[2]
    else
        return c[3]
    end
end


para = @with_kw (γ = 3.0, η = 0.7, r = 0.04, β = 0.98, ξ_nodes = 20, ϵ = range(0.0, 3.0, 5),
    T = T, μ = mort, init_t = 40, ρ = 0.97, σ = 0.02, asset = collect(range(0.0, 25.0, 15)), 
    work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.1, δ = [0.7, 0.024, -0.0002], φ_l = 0.5, θ_b = 0.0, κ = 2.0,
    aime = profile, plan = collect(1:4), ra = 65, τ = 0.12, lme = profile)
para = para()

v = zeros(length(para.asset), length(para.ϵ), length(para.wy), length(para.plan), length(para.work), length(para.aime), length(para.lme), para.T-para.init_t+2)
function solve(v::Array{Float64,8};para)
    #parameters
    (; γ, η, r, β, ξ_nodes, ϵ, T, μ, init_t, ρ, σ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme) = para
    #settings
    ξ, w = gausshermite(ξ_nodes)
    ξ = ξ |> x -> CuArray(x)
    w = w |> x -> CuArray(x)
    ϵ_grid = zeros(length(ϵ), T-init_t+2)
    ϵ_grid[:,1] = ϵ
    for t in 2:T-init_t+2
        ϵ_grid[:,t] = range(ρ*ϵ_grid[1,t-1] + sqrt(2)*σ*minimum(ξ), ρ*ϵ_grid[end,t-1] + sqrt(2)*σ*maximum(ξ), length(ϵ))
    end
    

    #policy function 1: asset, 2: work, 3: plan
    policy = zeros(length(asset), length(ϵ), length(wy), length(plan), length(work), length(aime), length(lme), T-init_t+1, 3)
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
    @tullio lumpsum_benefit[y,m,q] = (plan[q] == 2)*min(max(wy[m], 2*wy[m]-15), 50)*aime[y]
    @tullio monthly_benefit[y,m,q] = (plan[q] == 3)*max(wy[m]*aime[y]*0.00775+3, wy[m]*aime[y]*0.0155)
    #adjustment cost of working
    @tullio adj_cost[k,x] = (1-$φ_l*(work[k]-work[x] == 1))
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
    
    for s in tqdm(1:T-init_t+1)
        t = T-s+1
        wy_comp = δ[1] + δ[2]*t + δ[3]*t^2
        #extra_benefit should be pinned down    
        @tullio extra_benefit[q] = ex_benefit($t, $ra, plan[q])
        mort = μ[t]
        v_func = LinearInterpolation((asset, ϵ_grid[:,t-init_t+2], wy, plan, work, aime, lme), v[:,:,:,:,:,:,:,t-init_t+2])
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        
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
        @tullio consumption[i,j,l,m,k,x,q,y] = (1+$r)*asset[i] + income[l,m,k,q,y] + adj_cost[k,x] - asset[j]
        #leisure
        @tullio leisure[k] = (28-20*work[k])^(1-$η)
        #consumption floor
        @tullio utility[i,j,k,l,m,x,q,y] = consumption[i,j,l,m,k,x,q,y] ≥ $c_min ? (((consumption[i,j,l,m,k,x,q,y]^($η))*leisure[k])^(1-$γ))/(1-$γ) : -1e200
        #bequest
        @tullio bequest[j] = ($θ_b*($κ+asset[j])^(1-$γ))/(1-$γ)*$mort
        #forbidden path of applying for pension
        #1: unreceived, 2: lump-sum, 3: monthly, 4: received lump-sum
        @tullio forbid[p,q,m] = ((($t - $ra < -5)&&(plan[q] != 1))||((wy[m] == 0)&&((plan[q] != 1)||(plan[q] != 4)))||((plan[p] == 2)&&(plan[q] != 4))||((plan[p] == 3)&&(plan[q] != 3))||((plan[p] == 4)&&(plan[q] != 4)) ? -1e200 : 0.0)
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
        #try gaussian quadrature
        @tullio f_utility[j,k,l,m,y,z,q,h] = v_func(asset[j], f_ϵ[l,h], f_wy[m,k], plan[q], work[k], f_aime[y,z,l,m,k], f_lme[z,l,k])
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

sol = solve(v; para)

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
asset[1] = 4.0
work = zeros(para.T-para.init_t+2)
#initial working status
work[1] = 1
wy = zeros(para.T-para.init_t+2)
#initial work years
wy[1] = 7
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
#value goes wrong. check the punishment.
asset
