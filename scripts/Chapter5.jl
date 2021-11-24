using LTBonds 
using Plots
using BenchmarkTools
using LaTeXStrings 
using Random 
using PrettyTables

include("plotting_functions.jl")


penaltyAG06 = DefCosts(pen1 =0.02, pen2 = 0.0, quadratic = true, reentry = 0.0385)
penaltyCE12 = DefCosts(pen1 = -0.1881927550, pen2 = 0.2455843389, quadratic = true, reentry = 0.0385)
penaltyAr08 = DefCosts(pen1=0.0, pen2=0.0, threshold = 0.969, quadratic = false, reentry = 0.0385)
#Faster Re-entry
penaltyAG06Fast = DefCosts(pen1 =0.02, pen2 = 0.0, quadratic = true, reentry = 0.282)
penaltyCE12Fast = DefCosts(pen1 = -0.1881927550, pen2 = 0.2455843389, quadratic = true, reentry = 0.282)
penaltyAr08Fast = DefCosts(pen1=0.0, pen2=0.0,threshold = 0.969, quadratic = false, reentry = 0.282)


modelAG06, modelCE12, modelAr08, modelAG06Fast, modelCE12Fast, modelAr08Fast = map((penaltyAG06,penaltyCE12,penaltyAr08, penaltyAG06Fast,penaltyCE12Fast,penaltyAr08Fast)) do penalty
    R = 1.01
    β = 0.9540232420
    pref = Preferences(β = β, u = make_CRRA(ra = 2))
    y = discretize(YProcess(n = 200, ρ = 0.948503, std = 0.027092, μ = 0.0, span = 3.0, tails = false))
    m = MTruncatedNormal(; std =  10e-10, span = 2.0)
    bond = BondCE2012(n = 350, min = 0.0, max = 1.5, κ = 0.0, λ = 1.0)
    penalty = penalty
    generate_workspace(LTBondModel(
        y = y,
        m = m, 
        preferences = pref, 
        bond = bond, 
        def_costs = penalty, 
        R = R
    ))
end;


#Impatient Government
modelAG06IMP, modelCE12IMP, modelAr08IMP = map((penaltyAG06,penaltyCE12,penaltyAr08)) do penalty
    R = 1.01
    β = 0.8
    pref = Preferences(β = β, u = make_CRRA(ra = 2))
    y = discretize(YProcess(n = 200, ρ = 0.948503, std = 0.027092, μ = 0.0, span = 3.0, tails = false))
    m = MTruncatedNormal(; std =  10e-10, span = 2.0)
    bond = BondCE2012(n = 350, min = 0.0, max = 1.5, κ = 0.0, λ = 1.0)
    penalty = penalty
    generate_workspace(LTBondModel(
        y = y,
        m = m, 
        preferences = pref, 
        bond = bond, 
        def_costs = penalty, 
        R = R
    ))
end;


for m in (modelAG06, modelCE12, modelAr08, modelAG06Fast, modelCE12Fast, modelAr08Fast, modelAG06IMP, modelCE12IMP, modelAr08IMP)
    @time solve!(m; max_iters = 10000, g = 1.0, err = 1e-10, print_every = 50)
end 


function get_Bbars(m)
    bbar=[findfirst(get_d_pol(m)[:,i].==m.model.m.m_min) for i in 1:length(get_y_grid(m))]
    bbaridx=replace(bbar, nothing =>length(get_b_grid(m)))
    bbar=[get_b_grid(m)[x] for x in bbaridx]
    return(bbaridx',bbar')
end



############################
#Figures
############################

ygrid=get_y_grid(modelAG06)
get_ydef(ws)=ws.cache.cdef

#Figure 5.1: Endowment  in Default
f = plot(ygrid,ygrid,line=(lw45, :gray),legend=false, 
xlabel=(L"$y$"), ylabel=(L"$y^D$"))
plot!(f,ygrid,get_ydef(modelAG06),line=(lw, :black))
plot!(f,ygrid,get_ydef(modelCE12),line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(f,ygrid, get_ydef(modelAr08),line=(lw,:dashdot, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
savefig(f, (joinpath(@__DIR__,"..","output","Chapter5","fig_5_1.pdf" )))



# Figure 5.2: Deadweight Costs of Default
f = plot(ygrid, get_Bbars(modelAG06)[2][:],line=(lw,:black), xlabel=(L"$y$"),ylabel= (L"$\overline{b}(y)$"), legend=false);
plot!(ygrid, get_Bbars(modelCE12)[2][:],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(ygrid, get_Bbars(modelAr08)[2][:],line=(lw,:dashdot, :black), st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
savefig(f, (joinpath(@__DIR__,"..","output","Chapter5","fig_5_2.pdf" )))


# Figure 5.3:  Price Schedules
#plot prices at lowest Y (Figure 5.3 (a))
f=plot_q(1, modelAG06, modelCE12, modelAr08)
plot!(ylabel = (L"$q(b',y=y_{min})/q^{RF}$"))
savefig(f, (joinpath(@__DIR__,"..","output","Chapter5","fig_5_3a.pdf" )))

#plot prices at mean Y (Figure 5.3 (b))
midY = length(ygrid) ÷ 2
f=plot_q(midY, modelAG06, modelCE12, modelAr08)
plot!(ylabel = (L"$q(b',y=\mu)/q^{RF}$"))
savefig(f, (joinpath(@__DIR__,"..","output","Chapter5","fig_5_3b.pdf" )))


#plot prices at max Y (Figure 5.3 (c))
f=plot_q(length(ygrid),  modelAG06, modelCE12, modelAr08)
plot!(ylabel = (L"$q(b',y=y_{max})/q^{RF}$"))
savefig(f, (joinpath(@__DIR__,"..","output","Chapter5","fig_5_3c.pdf" )))


# Figure 5.4 Policy Functions
#plot policies at lowest Y (Figure 5.4 (a))
f=plot_pol(1, modelAG06, modelCE12, modelAr08,get_Bbars(modelAG06)[1][1],get_Bbars(modelCE12)[1][1],get_Bbars(modelCE12)[1][1])
plot!(ylabel = (L"$q(b',y=y_{min})/q^{RF}$"))
savefig(f, (joinpath(@__DIR__,"..","output","Chapter5","fig_5_4a.pdf" )))

#plot policies at mean Y (Figure 5.4 (b))
midY = length(ygrid) ÷ 2
f=plot_pol(midY, modelAG06, modelCE12, modelAr08,get_Bbars(modelAG06)[1][midY],get_Bbars(modelCE12)[1][midY],get_Bbars(modelCE12)[1][midY])
plot!(ylabel = (L"$q(b',y=\mu )/q^{RF}$"))
savefig(f, (joinpath(@__DIR__,"..","output","Chapter5","fig_5_4b.pdf" )))

#plot policies at max Y (Figure 5.4 (c))
f=plot_pol(length(ygrid),  modelAG06, modelCE12, modelAr08,get_Bbars(modelAG06)[1][end],get_Bbars(modelCE12)[1][end],get_Bbars(modelCE12)[1][end])
plot!(ylabel = (L"$q(b',y=y_{max})/q^{RF}$"))
savefig(f, (joinpath(@__DIR__,"..","output","Chapter5","fig_5_4c.pdf" )))




###simulations
big_T = 20_000 
big_N = 1_000
rng = Random.seed!(1234)

@time shocks, paths = create_shocks_paths(modelAG06, big_T, big_N; rng) 

moments_AG, moments_CE, moments_Ar =map((modelAG06, modelCE12, modelAr08)) do m
    @time simulation!(paths, shocks, m; n = big_T, trim = 1000, trim_def = 20)
    moments(paths, m)
end

#Table 5.2
pretty_table(
    [
        pairs(moments_AG),
        pairs(moments_CE),
        pairs(moments_Ar)
    ],
    row_names = ["Linear", "Quadratic", "Threshold"]
)


moments_AG, moments_CE, moments_Ar =map((modelAG06IMP, modelCE12IMP, modelAr08IMP)) do m
    @time simulation!(paths, shocks, m; n = big_T, trim = 1000, trim_def = 20)
    moments(paths, m)
end

#Table 5.3
pretty_table(
    [
        pairs(moments_AG),
        pairs(moments_CE),
        pairs(moments_Ar)
    ],
    row_names = ["Linear", "Quadratic", "Threshold"]
)


moments_AG, moments_CE, moments_Ar =map((modelAG06Fast, modelCE12Fast, modelAr08Fast)) do m
    @time simulation!(paths, shocks, m; n = big_T, trim = 1000, trim_def = 20)
    moments(paths, m)
end

#Table 5.4
pretty_table(
    [
        pairs(moments_AG),
        pairs(moments_CE),
        pairs(moments_Ar)
    ],
    row_names = ["Linear", "Quadratic", "Threshold"]
)

