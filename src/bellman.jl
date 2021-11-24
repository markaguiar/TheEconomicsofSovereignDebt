# Accessor functions used in the computations 
get_decay_b_at(a, bi) = get_decay(a)[bi]

get_Ev_at(a, biprime, y_state) = get_Ev(a)[biprime, y_state[1]]
get_vD_at(a, y_state) = get_vD(a)[y_state[1]]
get_b_pol_at(a,  bi_prime, y_state) = get_b_pol(a)[bi_prime, y_state...]
get_d_pol_at(a,  bi_prime, y_state) = get_d_pol(a)[bi_prime, y_state...]

set_d_pol_at!(a, bi, y_state,  val) = (get_d_pol(a)[bi, y_state...] = val)
set_v_new_at!(a, bi, y_state,  val) = (get_v_new(a)[bi, y_state...] = val)
set_bond_return_at!(a, bi, y_state, val) = (get_bond_return(a)[bi, y_state...] = val)
set_repay_at!(a, bi, y_state, val) = (get_repay(a)[bi, y_state...] = val)


find_bond_return_at(a, bi, biprime, yi) = _find_bond_return_at(get_bond(a), a, bi, biprime, yi) 

_find_bond_return_at(::AbstractBond, a, bi, biprime, (yi,))  = find_bond_return(a; q = get_q(a)[biprime, yi])

get_c_at(a, bi, bi_prime, y_state) = _get_c_at(get_bond(a), a, bi, bi_prime, y_state)

function _get_c_at(::AbstractBond, a, bi, bi_prime, (yi, ))
    q = get_q(a)
    b_grid = get_b_grid(a)
    y_grid = get_y_grid(a)
    get_c(a; y = y_grid[yi], b = b_grid[bi], q = q[bi_prime, yi], b_prime = b_grid[bi_prime])
end 


# Given a model m, workspace a, and exogenous state yi, finds the optimal debt 
# and default policy using a divide and conquer algorithm. 
function bellman_given_yi!(model, a, yi, ::DivideAndConquer)
    v = get_v(a)
    b_range = axes(v, 1)
    i_first = firstindex(b_range)
    i_last = lastindex(b_range)

    optimize!(model, a, yi, i_first, b_range)
    optimize!(model, a, yi, i_last, first(get_b_pol_at(a, i_first, yi)).idx:i_last)

    divide_and_conquer!(model, a, yi, i_first, i_last)
end


function divide_and_conquer!(model, a, yi, il, ih)
    if (ih - il > 1)
        b_pol_at_il = get_b_pol_at(a, il, yi)
        b_pol_at_ih = get_b_pol_at(a, ih, yi)

        imid = div(il + ih, 2)
        @inbounds b_range = first(b_pol_at_il).idx:last(b_pol_at_ih).idx
        optimize!(model, a, yi, imid, b_range)

        divide_and_conquer!(model, a, yi, il, imid)            
        divide_and_conquer!(model, a, yi, imid, ih)   
    end
    return nothing
end 


function optimize!(model, a, yi, bi, b_range)
    b_pol = get_b_pol(a) 
    @unpack vD, Ev = get_cache(a)

    @unpack m_min, m_max = get_m(a)
    @unpack β, u = get_preferences(a)

    b_pol = get_b_pol_at(a, bi, yi)

    pol_i_min, pol_i_max = 1, 1
    laffer_i = length(axes(Ev, 1))
    laffer_val = typemin(eltype(Ev))

    max_val_min = typemin(eltype(Ev))
    max_val_max = typemin(eltype(Ev))

    something_feasible_min = false
    something_feasible_max = false
    
    @inbounds for bi_prime in b_range
        c0 = get_c_at(a, bi, bi_prime, yi)
        c_m_min = c0 - m_min
        c_m_max = c0 - m_max

        if c0 > laffer_val
            laffer_val = c0
            laffer_i = bi_prime 
        end 

        # exploiting that c_max < c_min
        if c_m_min > 0  # checking feasibility 
            tmp = β * get_Ev_at(a, bi_prime, yi)
            something_feasible_min = true
            val = u(c_m_min) + tmp 
            if val > max_val_min
                max_val_min, pol_i_min = val, bi_prime 
            end

            if c_m_max > 0 
                something_feasible_max = true
                val = u(c_m_max) + tmp
                if val > max_val_max
                    max_val_max, pol_i_max = val, bi_prime 
                end                    
            end
        end 
    end 

    something_feasible_min || (pol_i_min = laffer_i)
    something_feasible_max || (pol_i_max = laffer_i)

    @inbounds empty!(b_pol)
    @inbounds push!(b_pol, PolicyPoint(idx = pol_i_min, m = m_min))
    @inbounds push!(b_pol, PolicyPoint(idx = pol_i_max, m = m_max))

    policy_thresholds!(model, a; bi, yi)  
    default_threshold!( model, a; bi, yi)
    integrate_v_and_bond_return!( model, a; bi, yi) 
end


function policy_thresholds!(model, a; bi, yi)
    m = get_m(model)
    @unpack β, u = get_preferences(model)
    p = get_b_pol_at(a, bi, yi)

    @inbounds begin         
        last_p = pop!(p)
        first_p = pop!(p)
        # At this point p is empty 

        l_min = first_p.idx
        l_max = last_p.idx

        # If policy at m_min != policy at m_max 
        if l_min != l_max
            l0 = l_min
            l1 = l0 + 1

            @inbounds while l1 <= l_max
                c0 = get_c_at(a, bi, l0, yi)
                c1 = get_c_at(a, bi, l1, yi)                 
                v0 = β * get_Ev_at(a, l0, yi)
                v1 = β * get_Ev_at(a, l1, yi)
                Δ = v0 - v1

                if c0 > c1 ||  Δ < eps() 
                    l1 = l1 + 1
                else
                    if (u(c1 - m.m_min) + v1 == u(c0 - m.m_min) + v0) && isempty(p) 
                        mstar = m.m_min 
                    else
                        mstar = find_m_root(a, c0, c1, Δ)
                    end 

                    if !isempty(p) && mstar < p[end].m
                        l0 = p[end].idx
                        pop!(p) 
                    else
                        push!(p, PolicyPoint(idx = l0, m = mstar))
                        l0 = l1
                        l1 = l0 + 1
                    end
                end
            end
        end
    end 
    push!(p, last_p)
end


function default_threshold!(model, a; bi, yi)
    @unpack β, u = get_preferences(model)
    @unpack m_min, m_max = get_m(model)

    @inbounds begin
        vD_ = get_vD_at(a, yi)
        pol = get_b_pol_at(a, bi, yi)
        
        c_base = get_c_at(a, bi, pol[1].idx, yi)
        ev = get_Ev_at(a, pol[1].idx, yi)
        c_m_min = c_base - m_min
        mD = m_max

        if c_m_min <= 0 || u(c_m_min) + β * ev < vD_
            mD = m_min 
        else
            c_m_max = get_c_at(a, bi, pol[end].idx, yi) - m_max
            ev_max = get_Ev_at(a, pol[end].idx, yi)
            if c_m_max <= 0 || u(c_m_max) + β * ev_max < vD_
                for p in pol 
                    biprime = p.idx
                    curr_m = p.m
                    c = get_c_at(a, bi, biprime, yi) - curr_m
                    ev_i = get_Ev_at(a, biprime, yi)
                    if c <= 0 || u(c) + β * ev_i < vD_
                        c0 = c + curr_m
                        mD = c0 - inv_u(u, vD_ - β * ev_i)
                        break 
                    end
                end 
            end 
        end         
        set_d_pol_at!(a, bi, yi, mD)
    end 
end


function integrate_v_helper(model, a, bi, biprime, yi, m0, m1)
    @unpack β, u = get_preferences(model)
    m = get_m(model)
    c = get_c_at(a, bi, biprime, yi)   
    eV = get_Ev_at(a, biprime, yi)
    m_mass = mass(m, m0, m1)
    v_m = integrate_u_c_minus_m(u, m, c, m0, m1) + β * eV * m_mass 
    bond_return_val = find_bond_return_at(a, bi, biprime, yi) * m_mass 
    return (v_m, bond_return_val)
end 


function integrate_v_and_bond_return!(model, a; bi, yi) 
    @unpack β, u = get_preferences(model)
    m = get_m(model)
    q = get_q(a)
    @unpack m_min, m_max = m

    @inbounds begin
        vD_val = get_vD_at(a, yi)
        pol = get_b_pol_at(a, bi, yi)
        mD = get_d_pol_at(a, bi, yi)

        if (mD <= m_min)  
            # always default 
            set_v_new_at!(a, bi, yi, vD_val) 
            set_bond_return_at!(a, bi, yi, zero(vD_val))
            set_repay_at!(a, bi, yi, zero(vD_val))
        else 
            v_m = zero(vD_val)
            bond_return_val = zero(eltype(q))
            m_lag = m_min
            for el in pol
                biprime = el.idx
                m_k = ifelse(el.m > mD, mD, el.m)                
                (m_k <= m_min) && continue   #  this has zero mass
                a1, a2 = integrate_v_helper(model, a, bi, biprime, yi, m_lag, m_k)
                v_m += a1 
                bond_return_val += a2
                (el.m >= mD) && break  # we are done here 
                m_lag = m_k
            end
            default_proba = survival(m, mD) 
            v_m += vD_val * default_proba
            set_v_new_at!(a, bi, yi, v_m)
            set_bond_return_at!(a, bi, yi, bond_return_val)
            set_repay_at!(a, bi, yi, 1 - default_proba)
        end
    end 
end

