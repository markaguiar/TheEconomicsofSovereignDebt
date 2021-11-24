get_c(b; kwargs...) = get_c(get_bond(b); kwargs...)
get_c(b::AbstractBond; kwargs...) = _get_c(has_maturing_coupon(b), b; kwargs...)

function _get_c(::NoMaturingCoupon, bond; y, b, q, b_prime) 
    λ, κ = get_λ(bond), get_κ(bond)
    return y + q * (b_prime - (1 - λ) * b) - (κ * (1 - λ) + λ) * b
end 

function _get_c(::MaturingCoupon, bond; y, b, q, b_prime) 
    λ, κ = get_λ(bond), get_κ(bond)    
    return y + q * (b_prime - (1 - λ) * b) - (κ + λ) * b
end 

risk_free_price(a::AbstractModel) = risk_free_price(get_bond(a); R = get_R(a))
risk_free_price(b::AbstractBond; kwargs...) = _risk_free_price(has_maturing_coupon(b), b; kwargs...)

function _risk_free_price(::NoMaturingCoupon, bond; R) 
    λ, κ = get_λ(bond), get_κ(bond)
    return (λ + (1 - λ) * κ) / (R - (1 - λ))
end 

function _risk_free_price(::MaturingCoupon, bond; R) 
    λ, κ = get_λ(bond), get_κ(bond)
    return (λ + κ) / (R - (1 - λ))
end 

find_bond_return(b; kwargs...) = find_bond_return(get_bond(b); kwargs...)
find_bond_return(b::AbstractBond; kwargs...) = _find_bond_return(has_maturing_coupon(b), b; kwargs...)

function _find_bond_return(::NoMaturingCoupon, bond; q) 
    λ, κ = get_λ(bond), get_κ(bond)
    return (λ + (1 - λ) * (κ + q))
end 

function _find_bond_return(::MaturingCoupon, bond; q) 
    λ, κ = get_λ(bond), get_κ(bond)
    return (λ + κ + (1 - λ) * q)
end


yield(b; kwargs...) = yield(get_bond(b); kwargs...)
yield(b::AbstractBond; kwargs...) = _yield(has_maturing_coupon(b), b; kwargs...)

function _yield(::NoMaturingCoupon, bond; q)
    λ, κ = get_λ(bond), get_κ(bond)
    return (λ + (1 - λ) * κ) / q - λ
end 

function _yield(::MaturingCoupon, bond; q)
    λ, κ = get_λ(bond), get_κ(bond)
    return (λ + κ) / q - λ
end 

