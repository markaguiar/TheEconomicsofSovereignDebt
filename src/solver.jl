
# Main solver -- iterates on prices and values until convergence or max_iters is reached. 
function solve!(m, a;
    method = DivideAndConquer(), max_iters = 10_000, g = 1.0, err = 1e-10, print_every = 50
)
    i = 0
    converged = false
    while true
        update_Ev!(m, a)
        update_all_vD!(m, a)
        update_v_and_policies!(m, a; method)
        update_q!(m, a)
        swap!(m, a, g)

        dis = distances(m, a)
        (max(dis...) < err) && (converged = true)

        if print_every != 0 && (mod(i, print_every) == 0 || (i + 1 >= max_iters)) || converged
            println("$(i+1): $dis")
        end
        converged && (println("Converged."); break)
    
        i += 1
        (i >= max_iters) && (@error("Did not converged"); break)
    end 
end 

solve!(a; kwargs...) = solve!(a.model, a; kwargs...)


# Updates the cotinuation under default 
function update_Ev!(_, a)  
    Ev, v = get_Ev(a), get_v(a)
    model = get_base_pars(a)
    mul!(Ev, v, get_y(model).Π)
end 


# Updates the default values  
function update_all_vD!(model, a)
    @unpack vD, Ev, ucdef, cdef = get_cache(a)
    vD1_new = get_new(a).vD1
    vD1 = get_current(a).vD1

    @unpack β, u = get_preferences(model)
    izero = get_zero_index(get_bond(model))
    θ = get_base_pars(model).def_costs.reentry
    Π = get_y(model).Π
    m_max = get_m(model).m_max

    # updates the next period default value 
    @batch for yi in eachindex(get_y_grid(model))
        @inbounds vD1_new[yi] = ucdef[yi] + β * θ * Ev[izero, yi] + β * (1 - θ) * dot(view(Π, :, yi), vD1)
    end

    # updates the current value -- note the role of the m-shock here
    @batch for yi in eachindex(get_y_grid(model))
        @inbounds vD[yi] = u(cdef[yi] - m_max) + β * θ * Ev[izero, yi] + β * (1 - θ) * dot(view(Π, :, yi), vD1_new)
    end
end 


# Main function that optimizes the bellman equation updating the values and policies 
function update_v_and_policies!(m, a; method = DivideAndConquer()) 
    @batch for yi in axes(get_v(a), 2)
        bellman_given_yi!(m, a, (yi,),  method)
    end
end 


# Updates the prices
function update_q!(model, a)
    q_new = get_new(a).q
    d_pol = get_d_pol(a)
    bond_return = get_bond_return(a)

    Π = get_y(model).Π
    m_min = get_m(model).m_min
    
    @batch for idx in CartesianIndices((axes(q_new, 1), axes(q_new, 2)))
        bi, yi = Tuple(idx)
        @inbounds begin 
            tmp = zero(eltype(q_new))
            for yiprime in axes(q_new, 2)
                mD = d_pol[bi, yiprime]
                if mD > m_min 
                    tmp += bond_return[bi, yiprime] * Π[yiprime, yi]
                end
            end
            q_new[bi, yi] = tmp / get_R(model)
        end 
    end
end


# Swap the old values for the new ones, and smooth with factor g
function swap!(m, a, g)
    cur = get_current(a)
    new = get_new(a)

    if g == 1 
        a.current, a.new = a.new, a.current
    else
        cur.q .= (1 - g) .* cur.q .+ g .* new.q
        cur.v .= (1 - g) .* cur.v .+ g .* new.v
        cur.vD1 .= (1 - g) .* cur.vD1 .+ g .* new.vD1
    end 
end 


function distances(m, a)
    cur, new = get_current(a), get_new(a)
    s1, s2, s3 = zero(eltype(cur.v)), zero(eltype(cur.q)), zero(eltype(cur.vD1))

    @tturbo for j in eachindex(cur.v, cur.q)
        s1 = max(s1, abs(cur.v[j] - new.v[j]))
        s2 = max(s2, abs(cur.q[j] - new.q[j]))
    end

    @tturbo for j in eachindex(cur.vD1, new.vD1)
        s3 = max(s3, abs(cur.vD1[j] - new.vD1[j]))
    end

    return (v = s1, q = s2, vD = s3)
end 

