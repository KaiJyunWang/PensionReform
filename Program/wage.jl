using DataFrames, GLM, Statistics, Random, CairoMakie
using Distributions, Optim, LinearAlgebra, Polynomials

# simulate data 
n = 10000
ρ = 0.9
σ_ν = 0.1
σ_1 = 1.0
β0 = 8.0
β1 = 0.15
β2 = -0.0014
age = rand(20:60, n)

# Error terms
e1 = σ_1 * randn(n)
e2 = ρ * e1 + σ_ν * randn(n)
e3 = ρ * e2 + σ_ν * randn(n)


log_wage1 = β0 .+ β1 * age .+ β2 * age.^2 .+ e1
log_wage2 = β0 .+ β1 * (age .+ 1) .+ β2 * (age .+ 1).^2 .+ e2
log_wage3 = β0 .+ β1 * (age .+ 2) .+ β2 * (age .+ 2).^2 .+ e3

df = DataFrame(id = vec(permutedims(repeat(1:n, 1, 3))), t = vec(repeat(1:3, n)), 
               age = vec(permutedims(hcat(age, age .+ 1, age .+ 2))), 
               log_wage = vec(permutedims(hcat(log_wage1, log_wage2, log_wage3))))



# Censor rule 
reservation(age) = 7.5 + log(age)

df.work = df.log_wage .> reservation.(df.age) .+ randn(3 * n)
df.observed_wage = df.work .* df.log_wage

# filter out observation with age ≥ 60
df = filter(r -> r.age ≤ 60, df)

# Set observation gap. First work = 0. No work and no work history = -1.
gd = groupby(df, :id)
function get_observation_gap(x)
    if all(.!x)
       return - ones(Int, length(x))
    else
       return [-ones(Int, findfirst(x)-1)..., 0, [i - findlast(x[1:i-1]) for i in findfirst(x)+1:length(x)]...]
    end
end

function get_last_observation(x, y)
    if all(iszero, x)
       return fill(missing, length(x))
    else
       return [fill(missing, findfirst(!iszero, x))..., [x[findlast(!iszero, x[1:i-1])] for i in findfirst(!iszero, x)+1:length(x)]...]
    end
end

df = combine(gd, All(), :work => get_observation_gap => :observation_gap, :observed_wage => get_last_observation => :last_observed_wage, :age => get_last_observation => :last_observed_age)

# MLE
function ll(θ; df)
    β = θ[1:3]
    γ = θ[4:44]
    ρ = θ[45]
    σ_1 = exp(θ[46])
    σ_ν = exp(θ[47])
    σ_η = exp(θ[48])
    # variance of the first observations
    first_σ_list = fill(σ_1, maximum(df.t))
    first_σ_list[2:end] = [sqrt(ρ^2 * first_σ_list[i-1]^2 + σ_ν^2) for i in 2:length(first_σ_list)]
    lagged_σ_list = sqrt.(cumsum(ρ.^(0:2:2 * (maximum(df.t) - 1)))) .* σ_ν
    
    # f 
    f = Polynomial(β)

    ll = 0.0
    for i in 1:nrow(df)
       if df.observation_gap[i] == -1 
           ll += logcdf(Normal(0, sqrt(first_σ_list[df.t[i]]^2 + σ_η^2)), f(df.age[i]) - γ[df.age[i] - 19])
       elseif df.observation_gap[i] == 0
           ll += logcdf(Normal(0, sqrt(first_σ_list[df.t[i]]^2 + σ_η^2)), f(df.age[i]) - γ[df.age[i] - 19]) +
                logpdf(Normal(0, first_σ_list[df.t[i]]), df.observed_wage[i] - f(df.age[i]))
       else
           if df.work[i] == false
              ll += logcdf(Normal(0, sqrt(σ_η^2 + lagged_σ_list[df.t[i]]^2)), f(df.age[i]) - γ[df.age[i] - 19] - ρ * (df.last_observed_wage[i] - f(df.age[i])))
           else
              ll += logcdf(Normal(0, σ_η), df.observed_wage[i] - γ[df.age[i] - 19]) + logpdf(Normal(0, lagged_σ_list[df.t[i]]), df.observed_wage[i] - f(df.age[i]) - ρ^df.observation_gap[i] * (df.last_observed_wage[i] - f(df.last_observed_age[i])))
           end
       end
    end
    return -ll
end

# educated guess 
reg = lm(@formula(observed_wage ~ age + age^2), filter(r -> r.work, df))
std(residuals(reg))
θ0 = [coef(reg)..., Polynomial(coef(reg)).(20:60)..., 0.9, log(std(residuals(reg))), log(0.1 * std(residuals(reg))), 0.0]

res = optimize(θ -> ll(θ; df = df), θ0, NewtonTrustRegion(), autodiff = :forward, Optim.Options(show_trace = true, iterations = 100))

β = res.minimizer[1:3]
γ = res.minimizer[4:44]
ρ = res.minimizer[45]
σ_1 = exp(res.minimizer[46])
σ_ν = exp(res.minimizer[47])
σ_η = exp(res.minimizer[48])

begin
    f = Figure(size = (800, 400))
    ax = Axis(f[1,1])
    lines!(ax, 20:60, Polynomial([β0, β1, β2]).(20:60), label = "True", color = :teal, linewidth = 2)
    lines!(ax, 20:60, 7.5 .+ log.(20:60), label = "Reservation wage", color = :orange, linewidth = 2)
    lines!(ax, 20:60, Polynomial(res.minimizer[1:3]).(20:60), label = "Estimated", color = :brown, linewidth = 2)
    lines!(ax, 20:60, γ, label = "Estimated γ", color = :purple, linewidth = 2)
    Legend(f[1,2], ax, framevisible = false)
    f
end