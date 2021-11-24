

#set linewidth for plots:
lw=2
lw45=.5
ms=3
msdiamond=5
default(size=(600,400),xtickfontsize=12,ytickfontsize=12,yguidefontsize=14,xguidefontsize=14)

#to set spacing of markers and other attributes
@recipe function f(::Type{Val{:samplemarkers}}, x, y, z; step = 10)
    n = length(y)
    sx, sy = x[1:step:n], y[1:step:n]
    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markershape --> :auto
    x := sx
    y := sy
end

function plot_q(idx,a1,a2,a3)
    #set linewidth for plots:
    lw = 2
    ms = 3
    default(size = (600,400), xtickfontsize = 12, ytickfontsize = 12, yguidefontsize = 14, xguidefontsize = 14)
    msdiamond = 5

    qstar1, qstar2, qstar3 = map(risk_free_price ∘ get_base_pars, (a1, a2, a3))
    b_grid1, b_grid2, b_grid3 = map(get_b_grid, (a1, a2, a3))
    q1, q2, q3 = map(get_q, (a1, a2, a3))

    f = plot(b_grid1, q1[:, idx]./qstar1, line = (lw, :black), legend = false, xlabel = (L"$b'$"))
    plot!(f, b_grid2, q2[:, idx]./qstar2, line = (lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize = ms)
    plot!(f, b_grid3, q3[:, idx]./qstar3, line = (lw,:dashdot, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize = msdiamond)

    plot!(f, xlims = (0, b_grid1[end]))
    return f
end


function plot_pol(idx,a1,a2,a3,bidx1,bidx2,bidx3)
    #set linewidth for plots:
    lw = 2
    ms = 3
    default(size = (600,400), xtickfontsize = 12, ytickfontsize = 12, yguidefontsize = 14, xguidefontsize = 14)
    msdiamond = 5

    b_grid1, b_grid2, b_grid3 = map(get_b_grid, (a1, a2, a3))
    b1, b2, b3 = map((a1, a2, a3)) do m
        bpol=get_b_pol(m)[:,idx]
        bpol=[get_b_grid(m)[bpol[i][1,1].idx] for i=1:length(get_b_grid(m))]
    end
    f = plot(b_grid1,b_grid1,line=(lw45, :gray),legend = false, xlabel = (L"$b$"))
    plot!(f,b_grid1[1:bidx1], b1[1:bidx1], line = (lw, :black))
    plot!(f, b_grid2[1:bidx2], b2[1:bidx2], line = (lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize = ms)
    plot!(f, b_grid3[1:bidx3], b3[1:bidx3], line = (lw,:dashdot, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize = msdiamond)

    plot!(f, xlims = (0, b_grid1[end]))
    return f
end
