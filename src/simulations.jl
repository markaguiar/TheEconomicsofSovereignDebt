draws(a::WorkSpace; kwargs...) = draws(a.model; kwargs...)

function draws(model; n::S, rng = nothing, seed = 1234) where (S<:Integer)
    #y shock

    #set seed for income shocks:
    if rng === nothing 
        rng = Random.seed!(seed)
    end 

    y_ind = zeros(S, n)
    ycdf = cumsum(get_y(model).Π, dims=1)
    
    y_st_dist = gen_st_y_dist(model; tol=1e-10, max_iter = 1000)
    y_st_dist_CDF = cumsum(y_st_dist)

    m_shock = get_m(model)

    #m shock
    m_dist = Distributions.TruncatedNormal(zero(m_shock.m_min),
                m_shock.std, m_shock.m_min, m_shock.m_max)

    m_sim = rand(rng, m_dist, n)
    re_entry = rand(rng, n) .< model.def_costs.reentry  

    y_rand = rand(rng, n)
    y_ind[1] = searchsortedfirst(y_st_dist_CDF, y_rand[1])
    for t in 2:n
        @views y_ind[t] = searchsortedfirst(ycdf[:, y_ind[t-1]], y_rand[t])
    end

    return ADraw(m_sim, re_entry, y_ind)
end 


#compute stationary distribution of income
function gen_st_y_dist(model; tol::F, max_iter::S) where{F<:Real,S<:Integer}

    old_dist = ones(size(get_y_grid(model), 1)) / size(get_y_grid(model), 1)
    new_dist = zeros(size(get_y_grid(model), 1))

    i_count = 1
    d_dist = 1.0 + tol

    while (i_count<=max_iter) && (d_dist>tol)
        mul!(new_dist, get_y(model).Π, old_dist)

        d_dist = zero(tol)
        @tturbo for j in eachindex(new_dist, old_dist)
            d_dist = max(d_dist, new_dist[j] - old_dist[j])
        end 

        old_dist, new_dist = new_dist, new_dist
        i_count += 1
    end
    defl_fac = sum(new_dist)^(-1)
    return new_dist * defl_fac
end


### helper threaded map ###
function tmap(f, lst)
    t = map(x->(Threads.@spawn f(x)), lst)
    map(fetch, t)
end

# If shocks is a vector, do a threaded map
simulation(shocks::Vector, a; kwargs...) = tmap(x -> simulation(x, a.model, a; kwargs...), shocks)
simulation(shocks_vec::ADraw, model, a; n, trim, trim_def) = simulation!(Path(model, n), shocks_vec, model, a; n, trim, trim_def)

simulation!(paths::Vector, shocks_vec::Vector, a; kwargs...) = tmap(x -> simulation!(x[1], x[2], a.model, a; kwargs...), zip(paths, shocks_vec))

function simulation!(path::Path, shocks::ADraw, model, a; n, trim, trim_def)
    @unpack re_entry = shocks
    @unpack c, y, m, b, bp, q, qb, κ, tb, r, def, in_def, y_ind, b_ind, bp_ind, no_def_duration, in_sample, in_def_sample = path
    
    zero_idx = get_bond(model).zero
  
    b_ind[1] = zero_idx
    in_def[1] = false
    no_def_duration[1] = 1

    state = init_simulation_state(model, y_ind[1])
    for t in 1:n
        y_ind[t] = shocks.y_ind[t]
        m[t] = shocks.m[t]

        state = next_state(model, state, y_ind[t])

        # Default in the previous period, check reentry and adjust b 
        if (t > 1) && in_def[t - 1]
            if re_entry[t] 
                in_def[t] = false
                b_ind[t] = zero_idx
            else
                in_def[t] = true
            end
        else 
            in_def[t] = false
        end

        b[t] = get_b_grid(model)[b_ind[t]]

        def[t] = false

        if (!in_def[t]) && is_a_default(a, b_ind[t], state, m[t])
            def[t] = true 
            in_def[t] = true
        end 

        if in_def[t]
            no_def_duration[t] = 0
            y[t] = get_cache(a).cdef[y_ind[t]]
            c[t] = y[t] - m[t]
            tb[t] = y[t] - c[t]
            bp_ind[t] = zero_idx
            bp[t] = get_b_grid(model)[bp_ind[t]]
            if t < n 
                b_ind[t+1] = zero_idx
            end 
        else
            no_def_duration[t] = t > 1 ? no_def_duration[t-1] + 1 : 1
            y[t] = get_y_grid(model)[y_ind[t]]
            b_pol_len = length(get_b_pol(a)[b_ind[t], state...])
            if b_pol_len == 1
                bp_ind[t] = get_b_pol(a)[b_ind[t], state...][1].idx
            else
                for ind in 1:b_pol_len
                    if m[t] < get_b_pol(a)[b_ind[t], state...][ind].m
                        bp_ind[t] = get_b_pol(a)[b_ind[t], state...][ind].idx
                        break
                    end
                end
            end
            assign_values_sims!((c, κ, r), model, a, b_ind[t], bp_ind[t], state, t)
            bp[t] = get_b_grid(model)[bp_ind[t]]
            tb[t] = y[t] - c[t]
            q[t] = get_q(a)[bp_ind[t], y_ind[t]]
            qb[t] = q[t] * bp[t]

            if t < n
                b_ind[t+1] = bp_ind[t]
            end
        end

        in_def_sample[t] = false 
        in_sample[t] = false
        if (t > trim + 1) && (no_def_duration[t-1] > trim_def - 1)
            if !in_def[t]
                in_sample[t] = true
            end
            in_def_sample[t] = true
        end 
    end
 
    return path 
end 


init_simulation_state(_, y) = (y,)

next_state(_, _, y) = (y,)


function assign_values_sims!((c, κ, r), model, a, b, bp, (y1, ), t)
    κ[t] = get_κ(get_bond(a))
    c[t] = get_c_at(a, b, bp, y1)
    r[t] = yield(model; q = get_q(a)[bp, y1])
end 


is_a_default(a, b, y, m) = get_d_pol(a)[b, y...] < m


moments(sim_path::Path, a) = moments([sim_path], a)

# This takes a vector of simulations paths
function moments(sim_paths::Vector, a)
    model = a.model
    R = get_R(model)

    in_sample = (x, j) -> x.in_sample[j]

    total = 0 
    @inbounds for path in sim_paths 
        for j in eachindex(path.y)
            path.in_sample[j] && (total += 1) 
        end 
    end 

    tmp_1 = Array{eltype(sim_paths[1].y)}(undef, total)    
    tmp_2 = similar(tmp_1)
    
    spread = flatten_assign!((x, j) -> (1 + x.r[j])^4 - R^4, tmp_1, sim_paths, in_sample)
    mean_spread = mean(spread)
    std_spread = sqrt(var(spread))

    y = flatten_assign!((x, j) -> log(x.y[j] - x.m[j]), tmp_2, sim_paths, in_sample)
    cor_r_y = cor(spread, y)

    b_y = flatten_assign!((x, j) -> x.bp[j] / (x.y[j] - x.m[j]), tmp_2, sim_paths,  in_sample)
    cor_r_b_y = cor(spread, b_y)
    mean_bp_y = mean(b_y)
    
    mv_y = flatten_assign!((x, j) -> x.qb[j] / (x.y[j] - x.m[j]), tmp_2, sim_paths, in_sample)
    mean_mv_y = mean(mv_y)

    κ = flatten_assign!((x, j) -> x.κ[j], tmp_2, sim_paths, in_sample)
    log_c = flatten_assign!((x, j) -> log(x.c[j]), tmp_2, sim_paths, in_sample)
    var_c = var(log_c)

    log_y = flatten_assign!((x, j) -> log(x.y[j] - x.m[j]), tmp_2, sim_paths, in_sample)
    var_y = var(log_y)
    std_c_y = sqrt(var_c/var_y)

    tb_y = flatten_assign!((x, j) -> x.tb[j]/(x.y[j] - x.m[j]), tmp_2, sim_paths,  in_sample)
    cor_r_tb = cor(tb_y, spread)

    log_y = flatten_assign!((x, j) -> log(x.y[j] - x.m[j]), spread, sim_paths, in_sample)
    cor_tb_y = cor(tb_y, log_y)

    tot = def = run = 0
    @inbounds for path in sim_paths 
        for j in eachindex(path.y)
            if path.in_def_sample[j]
                tot += 1 
                path.def[j] && (def += 1) 
            end 
        end 
    end 

    def_rate = 1 - (1 - def/tot)^4
    run_share = run / def
    
    moments = (;
        mean_bp_y,
        mean_mv_y,
        def_rate,
        mean_spread,
        std_spread,
        std_c_y,
        cor_tb_y,
        cor_r_y,
        cor_r_b_y,
        cor_r_tb 
        )

    return moments
end 


# Helper for moments
function flatten_assign!(f, x, ys, f_cond)
    i = 1
    @inbounds for y in ys 
        for j in 1:length(y.y) 
            if f_cond(y, j)
                x[i] = f(y, j)
                i += 1
            end 
        end
    end 
    return x
end  


function create_shocks_paths(a, big_T, big_N; rng = nothing)
    shocks = [draws(a; n = big_T, rng) for _ in 1:big_N]  #
    paths = tmap(x -> Path(a, big_T), 1:big_N) 
    return shocks, paths
end 