module Mortality
export life_ceil, mortality, life_dist
using LinearAlgebra

#mortality law from Thatcher(1999)
#μ(t) = α/(1+β*exp(-γ*t)) + κ
#x = [α, β, γ, κ]

#maximum lifespan
function life_ceil(x::Array{Float64,1})
    return floor(Int, ceil((log(1-x[4])+log(x[2])-log(x[1]+x[4]-1))/x[3]))
end

function mortality(x::Array{Float64,1})
    T = life_ceil(x)
    μ = ones(T)
    for t in 1:T-1
        μ[t] = x[1]/(1+x[2]*exp(-x[3]*t))+x[4]
    end
    return μ
end

#life distributions
function life_dist(x::Array{Float64,1}, T::Int)
    D = zeros(T, T) |> x -> LowerTriangular(x)
    for t in 1:T
        for s in t+1:T
            D[s,t] = (1 - sum(D[t:s-1,t]))*x[s]
        end
    end
    return D
end

end