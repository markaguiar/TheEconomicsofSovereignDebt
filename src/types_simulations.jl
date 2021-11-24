
Base.@kwdef struct Path{F<:Real, S<:Integer}
    y::Array{F,1}
    m::Array{F,1}
    c::Array{F,1}
    tb::Array{F,1}
    r::Array{F,1}
    b::Array{F,1}
    bp::Array{F,1}
    q::Array{F,1}
    qb::Array{F,1}
    κ::Array{F,1}
    def::Array{Bool,1}
    in_def::Array{Bool,1}
    y_ind::Array{S,1}
    b_ind::Array{S,1}
    bp_ind::Array{S,1}
    no_def_duration::Array{S,1}
    in_sample::Array{Bool,1}
    in_def_sample::Array{Bool,1}
    ck_default::Array{Bool, 1}
end


function Path(model, n)
    t = eltype(get_y_grid(model))
    
    y = Array{t}(undef, n)
    m = Array{t}(undef, n)
    bp = Array{t}(undef, n)
    b = Array{t}(undef, n)
    q = Array{t}(undef, n)
    qb = Array{t}(undef, n)
    κ = Array{t}(undef, n)
    c = Array{t}(undef, n)
    tb = Array{t}(undef, n)
    r = Array{t}(undef, n)
    def = Array{Bool}(undef, n)
    in_def = Array{Bool}(undef, n)
    no_def_duration = Array{Int}(undef, n)
    b_ind = Array{Int}(undef, n) 
    bp_ind = Array{Int}(undef, n) 
    y_ind = Array{Int}(undef, n)
    ck_default = Array{Bool}(undef, n)
    in_sample = Array{Bool}(undef, n)
    in_def_sample = Array{Bool}(undef, n)

    return Path(; 
        c, y, m, b, bp, q, qb, κ, tb, r, def, in_def, y_ind, b_ind, bp_ind, no_def_duration, in_sample, in_def_sample, ck_default
    )
end 

struct ADraw{T0, T1, T2}
    m::T0
    re_entry::T1
    y_ind::T2
end

