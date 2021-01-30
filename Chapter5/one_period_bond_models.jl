


include(joinpath(".","src","OnePeriodDebt_Methods.jl"))
include(joinpath(".","src","OnePeriodDebt_Simulations.jl"))
using LaTeXStrings
using Plots
using PrettyTables
pgfplotsx()


#Compute Twelve Models: Three Costs X Two Discount Factors and Three Costs X Two Reentry Rates
#Model 1:  Linear Costs as in Aguiar-Gopinath (2006) -- prefix "lin"
#Model 2:  Quadratic Costs as Chatterjee-Eyigungor (2012) -- prefix "quad"
#Model 3:  Threshold Costs as in Arellano 2008 -- prefix "ar"

#common asset grid
assetBounds=[-1.5,0.0]
assetN=350
assetGrid=collect(LinRange(assetBounds[1],assetBounds[2],assetN))

#common endowment process (based on CE12)
yN=200
midY=UInt32(ceil(yN/2))
yRho=0.948503
ySigma2=0.027092^2 
yMu=0.0
yStdSpan=3.0
#Where to put mass from y>|mu+n*sd|.  true=>put on extreme grid points, false=>reallocate evenly
#false follows CE12
yInflateEndpoints=false
yParams=ar1Params(yN,yRho,ySigma2,yMu,yStdSpan,yInflateEndpoints)

#common preference parameters
Gamma=2.0
#common interest rate
R=1.01
#common re-entry rate:
Theta=0.0385  #(from CE12) Benchmark
Theta2=0.282 #From AR08
#Patience
Beta=0.9540232420 #Benchmark
Beta2=0.80 #Impatient alternative

#Simulation specs:
simulN=1000
simulT=20000


#Default Cost Specification
Pen=false
arPen=true #turns on AR08 cost function
#Aguiar-Gopinath 2006 Linear Costs
linD0=0.02
linD1=0.0
#CE12 Quadratic
quadD0=-0.1881927550
quadD1=0.2455843389
#Arellano 2008
arD0=0.969
arD1=0.0


############################
#Compute Models
############################

#Compute Model 1: Linear Default Cost
@info "Computing Benchmark Linear Cost Model"
linSpec=onePeriodDebtSpec(Beta,Theta,Gamma,linD0,linD1,R,yParams,assetGrid,Pen);
linEval=makeEval(linSpec);
GC.gc()
@time vfiGOneStep!(linSpec,linEval,2.2e-16,2000);

#Compute Model 2: Quadratic Cost
@info "Computing Benchmark Quadratic Cost Model"
quadSpec=onePeriodDebtSpec(Beta,Theta,Gamma,quadD0,quadD1,R,yParams,assetGrid,Pen);
quadEval=makeEval(quadSpec);
GC.gc()
@time vfiGOneStep!(quadSpec,quadEval,2.2e-16,2000);

#Compute Model 3: Threshold Cost
@info "Computing Benchmark Threshold Cost Model"
arSpec=onePeriodDebtSpec(Beta,Theta,Gamma,arD0,arD1,R,yParams,assetGrid,arPen);
arEval=makeEval(arSpec);
GC.gc()
@time vfiGOneStep!(arSpec,arEval,2.2e-16,2000);


############################
#With Impatient Discount Factor
############################
#Compute Model 1: Linear Default Cost -- Impatient
@info "Computing Impatient Linear Cost Model"
linSpecImp=onePeriodDebtSpec(Beta2,Theta,Gamma,linD0,linD1,R,yParams,assetGrid,Pen);
linEvalImp=makeEval(linSpecImp);
GC.gc()
@time vfiGOneStep!(linSpecImp,linEvalImp,2.2e-16,2000);

#Compute Model 2: Quadratic -- Impatient
@info "Computing Impatient Quadratic Cost Model"
quadSpecImp=onePeriodDebtSpec(Beta2,Theta,Gamma,quadD0,quadD1,R,yParams,assetGrid,Pen);
quadEvalImp=makeEval(quadSpecImp);
GC.gc()
@time vfiGOneStep!(quadSpecImp,quadEvalImp,2.2e-16,2000);


#Compute Model 3: Threshold -- Impatient
@info "Computing Impatient Threshold Cost Model"
arSpecImp=onePeriodDebtSpec(Beta2,Theta,Gamma,arD0,arD1,R,yParams,assetGrid,arPen);
arEvalImp=makeEval(arSpecImp);
GC.gc()
@time vfiGOneStep!(arSpecImp,arEvalImp,2.2e-16,2000);

############################
#With Faster Reentry
############################

#Compute Model 1: Linear Default Cost -- Faster Re-entry
@info "Computing Fast Re-entry Linear Cost Model"
linSpecFast=onePeriodDebtSpec(Beta,Theta2,Gamma,linD0,linD1,R,yParams,assetGrid,Pen);
linEvalFast=makeEval(linSpecFast);
GC.gc()
@time vfiGOneStep!(linSpecFast,linEvalFast,2.2e-16,2000);

#Compute Model 2: Quadratic -- Faster Re-entry
@info "Computing Fast Re-entry Quadratic Cost Model"
quadSpecFast=onePeriodDebtSpec(Beta,Theta2,Gamma,quadD0,quadD1,R,yParams,assetGrid,Pen);
quadEvalFast=makeEval(quadSpecFast);
GC.gc()
@time vfiGOneStep!(quadSpecFast,quadEvalFast,2.2e-16,2000);


#Compute Model 3: Threshold -- Faster Re-entry
@info "Computing Impatient Threshold Cost Model"
arSpecFast=onePeriodDebtSpec(Beta,Theta2,Gamma,arD0,arD1,R,yParams,assetGrid,arPen);
arEvalFast=makeEval(arSpecFast);
GC.gc()
@time vfiGOneStep!(arSpecFast,arEvalFast,2.2e-16,2000);



############################
#Extract Default Thresholds
############################
linBbarsindex=[findfirst(linEval.pol.defaultGrid[:,i].==false) for i in 1:yN];
linBbars=-[linEval.aGrid[findfirst(linEval.pol.defaultGrid[:,i].==false)] for i in 1:yN];
quadBbarsindex=[findfirst(quadEval.pol.defaultGrid[:,i].==false) for i in 1:yN];
quadBbars=-[quadEval.aGrid[findfirst(quadEval.pol.defaultGrid[:,i].==false)] for i in 1:yN];
arBbarsindex=[findfirst(arEval.pol.defaultGrid[:,i].==false) for i in 1:yN];
arBbars=-[arEval.aGrid[findfirst(arEval.pol.defaultGrid[:,i].==false)] for i in 1:yN];



############################
#Figures
############################

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

#Figure 5.1: Endowment  in Default
f = plot(linEval.yGrid,linEval.yGrid,line=(lw45, :gray),legend=false, 
xlabel=(L"$y$"), ylabel=(L"$y^D$"))
plot!(f,linEval.yGrid,linEval.yDefGrid,line=(lw, :black))
plot!(quadEval.yGrid,quadEval.yDefGrid,line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(arEval.yGrid,arEval.yDefGrid,line=(lw,:dashdot, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
savefig(f, (joinpath(".","Chapter5","output","fig_5_1.pdf" )))

# Figure 5.2: Deadweight Costs of Default
f = plot(linEval.yGrid,linBbars,line=(lw,:black), xlabel=(L"$y$"),ylabel= (L"$\overline{b}(y)$"), legend=false);
plot!(f,quadEval.yGrid,quadBbars,line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(f,arEval.yGrid,arBbars,line=(lw,:dashdot, :black), st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
savefig(f, (joinpath(".","Chapter5","output","fig_5_2.pdf" )))


# Figure 5.3: Price Schedules
#Figure 5.3(a) Prices at lowest Y
f=plot(-linEval.aGrid,linEval.qGrid[:,1],line=(lw, :black), xlabel=(L"$b'$"),ylabel=(L"$q(b',y=y_{min})$"),legend=false)
plot!(f,-quadEval.aGrid,quadEval.qGrid[:,1],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(-arEval.aGrid,arEval.qGrid[:,1],line=(lw,:dashdot, :black), st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
plot!(f,xlims=(0, -assetGrid[1]))
savefig(f, (joinpath(".","Chapter5","output","fig_5_3a.pdf" )))



#Figure 5.3(b) Prices at mean Y
f=plot(-linEval.aGrid,linEval.qGrid[:,midY],line=(lw, :black), xlabel=(L"$b'$"),ylabel=(L"$q(b',y=\mu)$"),legend=false)
plot!(f,-quadEval.aGrid,quadEval.qGrid[:,midY],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(-arEval.aGrid,arEval.qGrid[:,midY],line=(lw,:dashdot, :black), st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
plot!(f,xlims=(0, -assetGrid[1]))
savefig(f, (joinpath(".","Chapter5","output","fig_5_3b.pdf" )))


#Figure 5.3(c) Prices at highest Y
f=plot(-linEval.aGrid,linEval.qGrid[:,yN],line=(lw, :black), xlabel=(L"$b'$"),ylabel=(L"$q(b',y=y_{max})$"),legend=false)
plot!(f,-quadEval.aGrid,quadEval.qGrid[:,yN],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(-arEval.aGrid,arEval.qGrid[:,yN],line=(lw,:dashdot, :black), st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
plot!(f,xlims=(0, -assetGrid[1]))
savefig(f, (joinpath(".","Chapter5","output","fig_5_3c.pdf" )))




# Figure 5.4 Policy Functions
# Figure 5.4(a)  Policies at lowest Y
f=plot(-linEval.aGrid[linBbarsindex[1]:assetN],-linEval.aGrid[linEval.pol.apRGrid[linBbarsindex[1]:assetN,1]],line=(lw, :black),xlabel=(L"$b$"),ylabel=(L"$\mathcal{B}(b,y=y_{min})$"), legend=false)
plot!(f,-quadEval.aGrid[quadBbarsindex[1]:assetN],-quadEval.aGrid[quadEval.pol.apRGrid[quadBbarsindex[1]:assetN,1]],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(f,-arEval.aGrid[arBbarsindex[1]:assetN],-arEval.aGrid[arEval.pol.apRGrid[arBbarsindex[1]:assetN,1]],line=line=(lw,:dashdot, :black), st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
plot!(f,-arEval.aGrid,-arEval.aGrid,line=(lw45, :gray))
plot!(f,xlim=(0, -assetGrid[1]))
savefig(f, (joinpath(".","Chapter5","output","fig_5_4a.pdf" )))


# Figure 5.4(b) Policies at mean Y
f=plot(-linEval.aGrid[linBbarsindex[midY]:assetN],-linEval.aGrid[linEval.pol.apRGrid[linBbarsindex[midY]:assetN,midY]],line=(lw, :black),xlabel=(L"$b$"),ylabel=(L"$\mathcal{B}(b,y=\mu)$"), legend=false)
plot!(f,-quadEval.aGrid[quadBbarsindex[midY]:assetN],-quadEval.aGrid[quadEval.pol.apRGrid[quadBbarsindex[midY]:assetN,midY]],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(f,-arEval.aGrid[arBbarsindex[midY]:assetN],-arEval.aGrid[arEval.pol.apRGrid[arBbarsindex[midY]:assetN,midY]],line=(lw,:dashdot, :black), st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
plot!(f,-arEval.aGrid,-arEval.aGrid,line=(lw45, :gray))
plot!(f,xlim=(0, -assetGrid[1]))
savefig(f, (joinpath(".","Chapter5","output","fig_5_4b.pdf" )))



# Figure 5.4(c) Policies at max Y
f=plot(-linEval.aGrid[linBbarsindex[yN]:assetN],-linEval.aGrid[linEval.pol.apRGrid[linBbarsindex[yN]:assetN,yN]],line=(lw, :black),xlabel=(L"$b$"),ylabel=(L"$\mathcal{B}(b,y=y_{max})$"), legend=false)
plot!(f,-quadEval.aGrid[quadBbarsindex[yN]:assetN],-quadEval.aGrid[quadEval.pol.apRGrid[quadBbarsindex[yN]:assetN,yN]],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=ms)
plot!(f,-arEval.aGrid[arBbarsindex[yN]:assetN],-arEval.aGrid[arEval.pol.apRGrid[arBbarsindex[yN]:assetN,yN]],line=(lw,:dashdot, :black), st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=msdiamond)
plot!(f,-arEval.aGrid,-arEval.aGrid,line=(lw45, :gray))
plot!(f,xlim=(0, -assetGrid[1]))
savefig(f, (joinpath(".","Chapter5","output","fig_5_4c.pdf" )))





############################
#Simulations
############################

rowlabels=["b'/y","mv/y","Def Rate","meanSpread","StDevSpread","StdDev(c)/StdDev(y)","corr(tb,y)","corr(spread,y)","corr(b'/y,spread)","corr(y-c,spread)"]
header=["Moment","Linear Cost", "Quadratic Cost", "Threshold Cost"]


Shocks=makeShockSequences(linSpec,linEval,simulT,simulN);

@info "Simulating Benchmark"
linPaths,linMoments=simulatePathsMC(linSpec,linEval,Shocks,simulN,1000,20,4.0);
quadPaths,quadMoments=simulatePathsMC(quadSpec,quadEval,Shocks,simulN,1000,20,4.0);
arPaths,arMoments=simulatePathsMC(arSpec,arEval,Shocks,simulN,1000,20,4.0);
println("#Table 5.2 Benchmark")
open(joinpath(".","Chapter5","output","Table5_2.txt"), "w") do f
    pretty_table(f,hcat(rowlabels,linMoments,quadMoments,arMoments),header, formatters=ft_printf("%5.4f",2:4))
end

@info "Simulating Impatient Gov't"
linPathsImp,linMomentsImp=simulatePathsMC(linSpecImp,linEvalImp,Shocks,simulN,1000,20,4.0);
quadPathsImp,quadMomentsImp=simulatePathsMC(quadSpecImp,quadEvalImp,Shocks,simulN,1000,20,4.0);
arPathsImp,arMomentsImp=simulatePathsMC(arSpecImp,arEvalImp,Shocks,simulN,1000,20,4.0);
println("#Table 5.3 Impatient Gov't")
open(joinpath(".","Chapter5","output","Table5_3.txt"), "w") do f
    pretty_table(f,hcat(rowlabels,linMomentsImp,quadMomentsImp,arMomentsImp),header, formatters=ft_printf("%5.4f",2:4))
end

@info "Simulating Fast Re-entry"
linPathsFast,linMomentsFast=simulatePathsMC(linSpecFast,linEvalFast,Shocks,simulN,1000,20,4.0);
quadPathsFast,quadMomentsFast=simulatePathsMC(quadSpecFast,quadEvalFast,Shocks,simulN,1000,20,4.0);
arPathsFast,arMomentsFast=simulatePathsMC(arSpecFast,arEvalFast,Shocks,simulN,1000,20,4.0);
println("#Table 5.4 Fast Re-entry")
open(joinpath(".","Chapter5","output","Table5_4.txt"), "w") do f
    pretty_table(f,hcat(rowlabels,linMomentsFast,quadMomentsFast,arMomentsFast),header, formatters=ft_printf("%5.4f",2:4))
end

