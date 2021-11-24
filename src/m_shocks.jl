
function quadrature!(m::MShock, f, ma, mb)
    @unpack nodes, weights = m
    val = zero(ma)
    mid = (mb + ma)/2
    dis = (mb - ma)/2
    @inbounds for i in eachindex(nodes, weights)
        node = nodes[i]
        weight = weights[i]
        val += f(node * dis + mid) * weight
    end 
    dis * val
end 


function integrate_u_c_minus_m(u, m::MShock, c, ma, mb)
    if mb <= ma 
        return zero(mb)
    else  
        return quadrature!(m, x -> u(c - x) * pdf(m, x), ma, mb)
    end 
end 


# We want a function to do the root finding algorithm
find_m_root(m, c0, c1, βΔ) = find_m_root(get_u(m), c0, c1, βΔ) 

function find_m_root(::CRRA2, c0, c1, βΔ)
    # quadratic version
    return (
        (c0 + c1) - sqrt(
            (c1 - c0)^2 + 4 * ((c1 - c0) / βΔ)
        )
    ) / 2
end 

find_m_root(::Log, c0, c1, βΔ) = (c1 - c0 * exp(βΔ)) /  (1 - exp(βΔ))

function find_m_root(u::AbstractUtility, c0, c1, βΔ)
    # generic version 
    # this version does not check whether consumption remains positive. 
    return find_zero(
            (m -> u(c0 - m) + βΔ - u(c1 - m), 
             m -> u_prime(u, c1 - m) - u_prime(u, c0 - m), 
             m -> - u_prime_prime(u, c1 - m) + u_prime_prime(u, c0 - m)
            ), 
            zero(c0), Roots.Halley())
end 
