using Distributions, LinearAlgebra, Plots, Interpolations
using Parameters, Random, Tables, Profile
using CUDA, CUDAKernels, KernelAbstractions, Tullio, Adapt
using BenchmarkTools

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

#profile of aime
profile = [2.747, 3.48, 4.58]

#pension benefit formula
#unit: month wage
#pension type: 1: no pension, 2: lump-sum pension, 3: monthly pension, 4: received pension
function benefit(aime::Float64, p::Int64, h::Int64, t::Int64, ra::Int64)
    if (p == 1) || (p == 4)
        return 0.0
    elseif p == 2
        return min(max(h, 2*h-15), 50)*aime
    else
        return max(h*aime*0.00775+3, h*aime*0.0155)*(1+0.04*min(abs(t-ra), 5)*sign(t-ra))
    end
end
#pension tax formula
function tax(aime::Float64, p::Int, τ::Float64 = 0.12)
    if p == 1
        return aime*0.2*τ
    else
        return 0.0
    end
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

function stair(x::Float64; c::Array{Float64,1} = c)
    if x < c[2]
        return c[1]
    elseif x < c[3]
        return c[2]
    else
        return c[3]
    end
end

para = @with_kw (γ = 1.5, η = 0.6, r = 0.02, β = 1/(1+r), ϵ = collect(range(0.0, 2.0, 11)), 
    T = 80, μ = mort.μ, init_t = 40, ρ = 0.97, σ = 0.1, ξ = σ*randn(250), asset = collect(range(0.0, 25.0, 26)), 
    work = [0,1], wy = collect(0:30), wy_ceil = 30, c_min = 0.1, δ = [0.05, -0.004], φ_l = 0.5, θ_b = 0.0, κ = 2.0,
    aime = profile, plan = collect(1:4), ra = 60, τ = 0.12, lme = profile)
para = para()

v = zeros(length(para.asset), length(para.ϵ), length(para.wy), length(para.plan), length(para.work), length(para.aime), length(para.lme), para.T-para.init_t+2)
function solve(v::Array{Float64,8};para)
    (; γ, η, r, β, ϵ, T, μ, init_t, ρ, σ, ξ, asset, work, wy, wy_ceil, c_min, δ, φ_l, θ_b, κ, aime, plan, ra, τ, lme) = para

    wy_comp = [δ[1]*min(wy_ceil, wy[m]) + δ[2]*min(wy_ceil, wy[m])^2 for m in 1:length(wy)] |> x -> CuArray(x)

    #policy function 1: asset, 2: work, 3: plan
    policy = zeros(length(asset), length(ϵ), length(wy), length(plan), length(work), length(aime), length(lme), T-init_t+1, 3)
    #computing matrices
    lumpsum_benefit = zeros(length(aime), length(wy), length(plan))
    monthly_benefit = zeros(length(aime), length(wy), length(plan))
    benefit = zeros(length(aime), length(wy), length(plan))
    adj_cost = zeros(length(work), length(work))
    f_ϵ = zeros(length(ϵ))
    wage = zeros(length(ϵ), length(wy))
    c_aime = zeros(length(ϵ), length(wy))
    pension_tax = zeros(length(ϵ), length(wy))
    income = zeros(length(ϵ), length(wy), length(work), length(plan), length(aime))
    consumption = zeros(length(asset), length(asset), length(ϵ), length(wy), length(work), length(work), length(plan), length(aime))
    leisure = zeros(length(work))
    utility = zeros(length(asset), length(asset), length(work), length(ϵ), length(wy), length(work), length(plan), length(aime))
    bequest = zeros(length(asset))
    forbid = zeros(length(plan), length(plan), length(wy))
    f_lme = zeros(length(lme), length(ϵ), length(wy), length(work))
    f_wy = zeros(length(wy), length(work))
    less_5y_f_aime = zeros(length(aime), length(ϵ), length(wy), length(work))
    more_5y_f_aime = zeros(length(aime), length(lme), length(ϵ), length(wy), length(work))
    f_aime = zeros(length(aime), length(lme), length(ϵ), length(wy), length(work))
    f_utility = zeros(length(asset), length(work), length(ϵ), length(wy), length(aime), length(lme), length(plan))
    candidate = zeros(length(asset), length(ϵ), length(wy), length(plan), length(work), length(aime), length(lme), length(asset), length(work), length(plan))

    ϵ_grid = zeros(length(ϵ), T-init_t+2)
    ϵ_grid[:,1] = ϵ
    for t in 2:T-init_t+2
        ϵ_grid[:,t] = range(ρ*ϵ_grid[1,t-1] + minimum(ξ), ρ*ϵ_grid[end,t-1] + maximum(ξ), length(ϵ))
    end

    #benefit schemes
    #2: lump-sum pension, 3: monthly pension
    @tullio lumpsum_benefit[y,m,q] = (plan[q] == 2)*min(max(wy[m], 2*wy[m]-15), 50)*aime[y]
    @tullio monthly_benefit[y,m,q] = (plan[q] == 3)*max(wy[m]*aime[y]*0.00775+3, wy[m]*aime[y]*0.0155)
    #adjustment cost of working
    @tullio adj_cost[k,x] = (1-$φ_l*(work[k]-work[x] == 1))
    
    for s in 1:T-init_t+1
        t = T-s+1
        extra_benefit = (t-ra) ≤ 5 ? 0.04*(t-ra) : 0.0
        mort = μ[t]
        v_func = LinearInterpolation((asset, ϵ_grid[:,t-init_t+2], wy, plan, work, aime, lme), v[:,:,:,:,:,:,:,t-init_t+2])
        ϵ_cua = CuArray(ϵ_grid[:,t-init_t+1])
        
        #future ϵ, avg. 
        @tullio f_ϵ[l] = $ρ * ϵ_cua[l]
        #minimum wage is 2.747
        @tullio wage[l,m] = max(2.747, exp(ϵ_cua[l] + wy_comp[m]))
        #current wage transformed into aime
        @tullio c_aime[l,m] = stair(wage[l,m]; c = aime)
        @tullio pension_tax[l,m] = c_aime[l,m]*0.2*$τ
        #benefit
        @tullio benefit[y,m,q] = lumpsum_benefit[y,m,q] + monthly_benefit[y,m,q]*(1+$extra_benefit)
        #net income
        @tullio income[l,m,k,q,y] = wage[l,m]*work[k] + benefit[y,m,q] - pension_tax[l,m]*work[k]*(plan[q] == 1)
        #consumption
        @tullio consumption[i,j,l,m,k,x,q,y] = (1+$r)*asset[i] + income[l,m,k,q,y] + adj_cost[k,x] - asset[j]
        @tullio leisure[k] = 364-260*work[k]
        #consumption floor
        @tullio utility[i,j,k,l,m,x,q,y] = consumption[i,j,l,m,k,x,q,y] > $c_min ? ($η*log(consumption[i,j,l,m,k,x,q,y]) + (1-$η)*log(leisure[k]))*(1-$γ) : -1e200
        #bequest
        @tullio bequest[j] = $θ_b*($κ+asset[j])^(1-$γ)*$mort
        #forbidden path of applying for pension
        #1: unreceived, 2: lump-sum, 3: monthly, 4: received lump-sum
        @tullio forbid[p,q,m] = (((wy[m] == 0)&&(p != 4))||((plan[p] == 2)&&(plan[q] != 4))||((plan[p] == 3)&&(plan[q] != 3))||((plan[p] == 4)&&(plan[q] != 4)) ? -1e200 : 0.0)
        #future lowest monthly wage
        @tullio f_lme[z,l,m,k] = min(4.58, max(lme[z], wage[l,m]*work[k]))
        #future work years
        @tullio f_wy[m,k] = min(wy[m]+work[k], $wy_ceil)
        #future aime
        @tullio less_5y_f_aime[y,l,m,k] = c_aime[l,m]*work[k]/(wy[m]+work[k]) + wy[m]/(wy[m]+work[k])*aime[y]
        @tullio more_5y_f_aime[y,z,l,m,k] = aime[y] + 0.2*max(0, c_aime[l,m]*work[k]-lme[z])
        @tullio f_aime[y,z,l,m,k] = (((wy[m] + work[k]) > 0) ? (wy[m] < 5 ? less_5y_f_aime[y,l,m,k] : more_5y_f_aime[y,z,l,m,k]) : 2.747)
        @tullio f_aime[y,z,l,m,k] = min(4.58, f_aime[y,z,l,m,k])
        #EV part
        @tullio f_utility[j,k,l,m,y,z,q] = mean(v_func.(asset[j], f_ϵ[l] .+ ξ, f_wy[m,k], plan[q], work[k], f_aime[y,z,l,m,k], f_lme[z,l,m,k]))

        @tullio candidate[i,l,m,p,x,y,z,j,k,q] = utility[i,j,k,l,m,x,q,y] + (1-$mort)*$β*f_utility[j,k,l,m,y,z,q] + bequest[j] + forbid[p,q,m]
        
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
    
    return (policy = policy, ϵ_grid = ϵ_grid)
end

sol = solve(v; para)

#simulation
ϵ = zeros(para.T-para.init_t+1)
#initial ϵ
ϵ[1] = 0.7
for t in 2:para.T-para.init_t+1
    ϵ[t] = para.ρ*ϵ[t-1] + para.σ*randn()
end
asset = zeros(para.T-para.init_t+2)
#initial asset
asset[1] = 1.0
work = zeros(para.T-para.init_t+2)
#initial working status
work[1] = 1
wy = zeros(para.T-para.init_t+1)
#initial work years
wy[1] = 5
plan = zeros(para.T-para.init_t+2)
#initial pension status
plan[1] = 1
s_aime = zeros(para.T-para.init_t+1)
#initial aime
s_aime[1] = 3.0
lme = zeros(para.T-para.init_t+1)
#initial lme
lme[1] = 2.8

#wage
wage = zeros(para.T-para.init_t+1)
consumption = zeros(para.T-para.init_t+1)

for t in 1:para.T-para.init_t
    a_func = LinearInterpolation((para.asset, sol.ϵ_grid[:,t], para.wy, para.plan, para.work, para.aime, para.lme), sol.policy[:,:,:,:,:,:,:,t,1])
    w_func = LinearInterpolation((para.asset, sol.ϵ_grid[:,t], para.wy, para.plan, para.work, para.aime, para.lme), sol.policy[:,:,:,:,:,:,:,t,2])
    p_func = LinearInterpolation((para.asset, sol.ϵ_grid[:,t], para.wy, para.plan, para.work, para.aime, para.lme), sol.policy[:,:,:,:,:,:,:,t,3])
    asset[t+1] = a_func(asset[t], ϵ[t], wy[t], plan[t], work[t], s_aime[t], lme[t])
    work[t+1] = round(Int, w_func(asset[t], ϵ[t], wy[t], plan[t], work[t], s_aime[t], lme[t]))
    plan[t+1] = round(Int, p_func(asset[t], ϵ[t], wy[t], plan[t], work[t], s_aime[t], lme[t]))
    wage[t] = max(2.747, exp(ϵ[t] + wy[t]))
    #state transition
    wy_comp = para.δ[1]*wy[t] + para.δ[2]*wy[t]^2
    wage[t] = min(exp(ϵ[t] + wy_comp), 2.747)
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
plt = plot(age, wage, label = "wage", xlabel = "age", title = "wage profile")
plot!(age, asset[2:end], label = "asset")
vcat(findall(x -> x == 0, work[2:end]) .+ para.init_t .- 0.5, findall(x -> x == 0, work[2:end]) .+ para.init_t .+ 0.5) |> x -> sort(x) |> x -> vspan!(plt, x, color = :gray, label = "retirement", alpha = 0.3)


