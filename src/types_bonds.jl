
abstract type AbstractBond end 

#This bond pays a coupon at maturity
struct Bond{F<:Real, N<:Integer, T1, T2} <: AbstractBond
    min::F
    max::F
    n::N
    κ::F  # coupon rate
    λ::F  # inverse maturity
    grid::T1 
    zero::T2 #location of zero debt
end


#This bond does not pay a coupon at maturity
struct BondCE2012{F<:Real, N<:Integer, T1, T2} <: AbstractBond
    min::F
    max::F
    n::N
    κ::F  # coupon rate
    λ::F  # inverse maturity
    grid::T1 
    zero::T2 #location of zero debt
end

get_λ(b::AbstractBond) = b.λ
get_grid(b::AbstractBond) = b.grid
get_zero_index(b::AbstractBond) = b.zero

get_κ(b::AbstractBond) = b.κ

function _bond_constructor(;min, max, n, κ, λ)
    grid = collect(LinRange(min, max, n))
    # making sure that 0 is on the grid
    zeroidx = findfirst(x -> x >= 0, grid)
    grid[zeroidx] = zero(eltype(grid))
    return (min, max, n, κ, λ, grid, zeroidx)
end 

Bond(;min, max, n, κ, λ) = Bond(_bond_constructor(;min, max, n, κ, λ)...)
BondCE2012(;min, max, n, κ, λ) = BondCE2012(_bond_constructor(;min, max, n, κ, λ)...)

# Maturing coupon traits 

abstract type FinalCoupon end 

struct MaturingCoupon <: FinalCoupon end 
struct NoMaturingCoupon <: FinalCoupon end 

has_maturing_coupon(::Bond) = MaturingCoupon()
has_maturing_coupon(::BondCE2012) = NoMaturingCoupon()
