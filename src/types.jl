
Base.@kwdef struct YProcess{F<:Real, N<:Integer}
    n::N
    ρ::F
    std::F
    μ::F
    span::F
    tails::Bool
end


Base.@kwdef struct YDiscretized{T0<:YProcess, T1, T2, T3}
    process::T0  # original process
    grid::T1
    Π::T2
    lr_mean::T3
end


abstract type MShock end 

get_d(m::MShock) = m.d


struct MTruncatedNormal{R<:Real, S, T} <: MShock
    d::S
    m_min::R
    m_max::R
    std::R

    # probability calculations 
    pdf_adjust_factor::R
    cdf_1::R
    cdf_2::R
    cdf_3::R
    # quadrature parameters
    nodes::T
    weights::T
    tmp::T
end

function MTruncatedNormal(; std, span, quadN = 100) 
    m_min = -span * std
    m_max = - m_min
    nodes, weights = gausslegendre(quadN)
    nodes_tmp = similar(nodes)
    d = Normal(zero(m_min), std)
    pdf_adjust_factor =  invsqrt2π / ((cdf(d, m_max) - cdf(d, m_min)) * std)
    cdf_1 = invsqrt2 / std
    cdf_2 = cdf(d, m_min)
    cdf_3 = 1 / ((cdf(d, m_max) - cdf(d, m_min)))

    return MTruncatedNormal(
        truncated(d, m_min, m_max),
        m_min, 
        m_max, 
        std,
        pdf_adjust_factor,
        cdf_1,
        cdf_2,
        cdf_3, 
        nodes, 
        weights, 
        nodes_tmp
    )
end 

## Preference types 


abstract type AbstractUtility end 

struct Log <: AbstractUtility end 

struct Power{T1, T2}
    power::T1 # exponent
    inv::T2 # inverse exponent
end

Power(x::Real) = Power(convert(AbstractFloat, x), convert(AbstractFloat, 1/x))
get_power(u::Power) = u.power


struct CRRA2 <: AbstractUtility end 

struct CRRA{T1, T2} <: AbstractUtility 
    uf::T1
    ra::T2
end

function CRRA(ra) 
    @assert ra != 1
    CRRA(Power(1 - ra), ra)
end 

get_power(u::CRRA) = get_power(u.uf)



Base.show(io::IO, m::CRRA) = print(io, "CRRA(ra = $(m.ra))")
Base.show(io::IO, ::Log) = print(io, "Log")
Base.show(io::IO, ::CRRA2) = print(io, "CRRA2")

function make_CRRA(; ra = 2)
    u = ra == 1 ? Log() : 
        ra == 2 ? CRRA2() : 
        CRRA(Power(1 - ra), ra)
    return u
end

make_CRRA(ra) = make_CRRA(; ra = ra)


Base.@kwdef struct Preferences{T<:AbstractUtility, F<:Real}
    β::F
    u::T
end


Base.@kwdef struct DefCosts{F<:Real}
    pen1::F
    pen2::F
    threshold::F = 0.0
    quadratic::Bool
    reentry::F
end


# Auxiliary struct
Base.@kwdef struct PolicyPoint{T1, T2}
    idx::T1
    m::T2
end 


# types for solver methods

abstract type AbstractSolverMethod end 

struct FasterDivideAndConquer <: AbstractSolverMethod end 

struct DivideAndConquer <:AbstractSolverMethod end 

struct SimpleMonotone <: AbstractSolverMethod end 

struct Sequential <: AbstractSolverMethod end 

