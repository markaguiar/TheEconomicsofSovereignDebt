
init_v(model::AbstractModel{B}) where B<:AbstractBond = [get_u(model)(y) /(1 - get_preferences(model).β) for _ in get_b_grid(model), y in get_y_grid(model)]

#init_v(model::AbstractModel{B}) where B<:AbstractFloatingRateBond = [get_u(model)(y) /(1 - get_preferences(model).β) for _ in get_b_grid(model), y in get_y_grid(model), _ in get_y_grid(model)]


function generate_workspace(
    model; 
    v = nothing,
    vD1 = nothing, 
    q = nothing, 
    κ = nothing
)

    @unpack pen1, pen2, threshold, quadratic = get_base_pars(model).def_costs
    @unpack β, u = get_preferences(model)

    y_grid = get_y_grid(model)
    b_grid = get_b_grid(model)
    threshold=threshold*model.y.lr_mean  #changed this


    m = get_m(model) 
    bond = get_bond(model)
    λ = get_λ(bond) 

    if v === nothing 
        v = init_v(model)
    end 

    if q === nothing 
        q = fill(risk_free_price(model), length(b_grid), length(y_grid))
    end

    if κ === nothing 
        κ = fill(get_R(model) - 1, length(b_grid), length(y_grid))
    end

    b_pol = map(_ -> begin 
            h = Vector{PolicyPoint{eltype(axes(v, 1)), typeof(m.m_min)}}()
            sizehint!(h, 15)
            return h
        end, v) # initializing the debt policy matrix

    cdef = quadratic ? y_grid .- max.(pen1 .* y_grid .+ pen2 .* y_grid.^2, 0) : min.(y_grid, threshold)
    ucdef = [integrate_u_c_minus_m(u, m, c, m.m_min, m.m_max) for c in cdef]
    
    if vD1 === nothing 
        vD1 = ucdef ./ (1 - β)
    else 
        vD1 = vD1 
    end 
    
    d_pol = similar(v)  # default policy matrix: m cutoffs for default

    vD = deepcopy(vD1)
    tmp_vD = similar(vD1)
    Ev = similar(q)
    bond_return = zero(v)
    
    cur = Iterables(v, vD1, q, κ)
    new = deepcopy(cur)

#    policies = Policies(b_pol, d_pol, deepcopy(d_pol))
    policies = Policies(b_pol, d_pol)
    
    decay_b = [findlast(x -> x <= (1 - λ) * b, b_grid) for b in b_grid]
    repay = zero(cur.v)
    
    cache = Cache(vD, Ev, cdef, ucdef, tmp_vD, bond_return, decay_b, repay)

    return WorkSpace(model, cur, new, policies, cache)
end 

