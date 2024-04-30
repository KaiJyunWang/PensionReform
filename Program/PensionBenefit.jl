module PensionBenefit
export benefit, tax

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

end