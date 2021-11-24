# -*- coding: utf-8 -*-
# # Chapter 7 Figures and Moments

import Pkg; 
Pkg.activate(joinpath(@__DIR__, "..")); 
Pkg.instantiate()

using LTBonds 
using Plots
using LaTeXStrings 
using Random 
using PrettyTables

pretty_table_output = :html # change to :text if running from terminal 

include("plotting_functions.jl")

modelLB, modelSB, modelLB2 = map((0.05, 1.0, 0.025)) do λ
    R = 1.01
    β = 0.9540232420
    pref = Preferences(β = β, u = make_CRRA(ra = 2))
    y = discretize(YProcess(n = 200, ρ = 0.948503, std = 0.027092, μ = 0.0, span = 3.0, tails = false))
    m = MTruncatedNormal(; std = (λ < 0.05 ? 0.005 : 0.003), span = 2.0)
    bond = BondCE2012(n = 350, min = 0.0, max = 1.5, κ = 0.03, λ = λ)
    penalty = DefCosts(pen1 = -0.1881927550, pen2 = 0.2455843389, quadratic = true, reentry = 0.0385)
    generate_workspace(LTBondModel(
        y = y,
        m = m, 
        preferences = pref, 
        bond = bond, 
        def_costs = penalty, 
        R = R
    ))
end;

for m in (modelLB, modelSB, modelLB2)
    @time solve!(m; max_iters = 10000, g = 0.5, err = 1e-10, print_every = 50)
end 


# ## Figures


#plot prices at lowest Y (Figure 7-6 (a))
f = plot_q(1, modelLB, modelSB, modelLB2)
savefig(f, (joinpath(@__DIR__,"..","output","Chapter7","fig_7_6a.pdf" )))
f

#plot prices at mean Y (Figure 7-6 (b))
midY = length(get_y_grid(modelLB)) ÷ 2
f = plot_q(midY, modelLB, modelSB, modelLB2)
savefig(f, (joinpath(@__DIR__,"..","output","Chapter7","fig_7_6b.pdf" )))
f

#plot prices at max Y (Figure 7-6 (c))
f = plot_q(length(get_y_grid(modelLB)), modelLB, modelSB, modelLB2)
savefig(f, (joinpath(@__DIR__,"..","output","Chapter7","fig_7_6c.pdf" )))
f

###simulations
big_T = 20_000 
big_N = 1_000
rng = Random.seed!(1234)

@time shocks, paths = create_shocks_paths(modelLB, big_T, big_N; rng) 
@time simulation!(paths, shocks, modelLB; n = big_T, trim = 1000, trim_def = 20)
@time moments_LB = moments(paths, modelLB)
pretty_table(
    [
        pairs(moments_LB),
    ],
    row_names = ["EGLB"],
    backend = Val(pretty_table_output)
)

moments_LB, moments_SB, moments_LB2 =map((modelLB, modelSB, modelLB2)) do m
    @time simulation!(paths, shocks, m; n = big_T, trim = 1000, trim_def = 20)
    moments(paths, m)
    end;

pretty_table(
    [
        pairs(moments_SB),
        pairs(moments_LB),
        pairs(moments_LB2)
    ],
    row_names = ["SB", "LB", "LB2"],
    backend = Val(pretty_table_output)
)


