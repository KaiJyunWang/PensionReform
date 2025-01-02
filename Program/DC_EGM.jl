using Interpolations, FastGaussQuadrature, Distributions
using CUDA, CUDAKernels, KernelAbstractions, Tullio, CUDA.Adapt
using Parameters
using CairoMakie

# Life-Cycle with DC-EGM method

# Parameters 
function set_parameters(; γ = 3f0, β = 0.96f0, R = 1f0/0.96f0, σ = 1.0f0, σ_ξ = 0.3f0, 
    ρ = 0.97f0, η = 0.9f0, φ_1 = 0.5f0, φ_2 = 3.0f0, T = 100, θ_b = 0.0f0, κ = 700.0f0)

    # Grids
    a_grid = collect(range(1e-3, 300.0, length = 201)) |> cu
    ϵ_grid = collect(range(-3*σ_ξ/sqrt(1-ρ^2), 3*σ_ξ/sqrt(1-ρ^2), length = 10)) |> cu
    n_grid = collect(0.0:1.0) |> cu
    ξ, w = gausshermite(10)
    ξ = cu(ξ) 
    w = cu(w)

    state_a = repeat(a_grid, 1, length(ϵ_grid), length(n_grid), length(ξ)) |> x -> permutedims(x, [1, 2, 3, 4]) |> cu 
    state_ϵ = repeat(ϵ_grid, 1, length(a_grid), length(n_grid), length(ξ)) |> x -> permutedims(x, [2, 1, 3, 4]) |> cu
    state_n = repeat(n_grid, 1, length(a_grid), length(ϵ_grid), length(ξ)) |> x -> permutedims(x, [2, 3, 1, 4]) |> cu
    state_ξ = repeat(ξ, 1, length(a_grid), length(ϵ_grid), length(n_grid)) |> x -> permutedims(x, [2, 3, 4, 1]) |> cu

    # Utility 
    u(c,n) = γ == 1 ? η*log(c) + (1-η)*log(1 - φ_1*n) : (c^η*(1 - φ_1*n)^(1-η))^(1-γ)/(1-γ) 
    u_prime(c,n) = η*(1-φ_1*n)^((1-γ)*(1-η))*c^(-γ*η+η-1) 
    u_prime_inv(u,n) = (u/(η*(1-φ_1*n)^((1-γ)*(1-η))))^(1/(-γ*η+η-1))

    return (γ = γ, β = β, R = R, σ = σ, σ_ξ = σ_ξ, ρ = ρ, η = η, φ_1 = φ_1, φ_2 = φ_2, T = T, θ_b = θ_b, κ = κ, 
        a_grid = a_grid, ϵ_grid = ϵ_grid, n_grid = n_grid, ξ = ξ, w = w, 
        state_a = state_a, state_ϵ = state_ϵ, state_n = state_n, state_ξ = state_ξ,
        u = u, u_prime = u_prime, u_prime_inv = u_prime_inv)
end

p = set_parameters()

function egm(policy_a, policy_n, mort; p) 
    # Parameters 
    @unpack γ, β, R, σ, σ_ξ, ρ, η, φ_1, φ_2, T, θ_b, κ, a_grid, ϵ_grid, n_grid, ξ, w, state_a, state_ϵ, state_n, state_ξ, u, u_prime, u_prime_inv = p
    # EGM step
    a_func = extrapolate(interpolate((a_grid, ϵ_grid, n_grid), policy_a, Gridded(Linear())), Linear()) |> cu
    n_func = extrapolate(interpolate((a_grid, ϵ_grid, n_grid), policy_n, Gridded(Constant())), Flat()) |> cu 

    next_period_ϵ = ρ * state_ϵ + sqrt(2) * σ_ξ * state_ξ
    next_period_n = n_func.(state_a, next_period_ϵ, state_n)
    next_period_a = a_func.(state_a, next_period_ϵ, state_n)

    @tullio work_cost[i,j,l,k] := (next_period_n[i,j,l,k] - n_grid[l] == 1) * $φ_2
    @tullio future_disposable[i,j,l,k] := $R * a_grid[i] + exp(next_period_ϵ[i,j,l,k]) * next_period_n[i,j,l,k] - work_cost[i,j,l,k]
    @tullio future_c[i,j,l,k] := future_disposable[i,j,l,k]*(1 - next_period_a[i,j,l,k])
    
    @tullio conti_u_prime[i,j,l,k] := $β * $R * u_prime(future_c[i,j,l,k], next_period_n[i,j,l,k]) 
    @tullio Eu_prime[i,j,l] := (1-$mort) * conti_u_prime[i,j,l,k]*w[k]/sqrt(π)
    @tullio bequest[i] := $β * $θ_b * ($κ + a_grid[i])^(-$γ) * $mort
    @tullio current_c[i,j,l] := u_prime_inv(Eu_prime[i,j,l], n_grid[l])
    @tullio current_disposable[i,j,l] := a_grid[i] + current_c[i,j,l]

    # Compute current grids 
    @tullio current_a_grid[i,j,l,m] := (current_disposable[i,j,l] - exp(ϵ_grid[j]) * n_grid[l] - $φ_2*(n_grid[l] - n_grid[m] == 1))/$R
    return current_c
end

policy_a = zeros(length(p.a_grid), length(p.ϵ_grid), length(p.n_grid))
policy_n = zeros(length(p.a_grid), length(p.ϵ_grid), length(p.n_grid))

res = egm(policy_a, policy_n, 0.01; p = p)
