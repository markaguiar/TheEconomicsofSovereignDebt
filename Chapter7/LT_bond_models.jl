
include(joinpath(".","src","LongTermDebt_Methods.jl"))
include(joinpath(".","src","LongTermDebt_Simulations.jl"))
using LaTeXStrings 
using Plots
using PrettyTables
pgfplotsx()


#The following are exactly the parameters of Chatterjee and Eyigungor (2012)
#The only exception is extending the asset bound to -1.5 from -1.0
beta=0.9540232420
theta=0.0385
gamma=2.0 #code is written assuming gamma>1
d0=-0.1881927550
d1=0.2455843389
R1=1.01
Lambda=0.05
coupon=0.03
assetBounds=[-1.5,0.0]
assetN=350
assetGrid=collect(LinRange(assetBounds[1],assetBounds[2],assetN))

#Arellano 2008's threshold default penalty true/false:
arellanoPenalty=false

#Process for endowment
yN=200 #number of gridpoints
rhoY=0.948503 #AR(1) coefficient
sigma2Y=0.027092^2 #variance of innovation
muY=0.0 #mean
yGridSpan=3.0 #span of discretized grid

#Where to put mass from y>|mu+n*sd|.  true=>put on extreme grid points, false=>reallocate evenly
yInflateEndpoints=false

#iid Endowment 
mPoints=12
sigma2m=0.003^2
mum=0.0
mGridSpan=2.0

#smoothing parameters governing updating price schedules and value functions (0 is full step)
smoothQ=0.5
smoothV=0.5


#fill in structure of endowment parameters:
YParams=ar1Params(yN,rhoY,sigma2Y,muY,yGridSpan,yInflateEndpoints)
mParams=iidParams(mPoints,sigma2m,mum,mGridSpan)

#fill in structure for model parameters:
LTBSpec=longTermBondSpec(beta,theta,gamma,d0,d1,R1,Lambda,coupon,assetBounds,assetN,YParams,mParams,arellanoPenalty,smoothQ,smoothV);
#redo with one-period bonds
STBSpec=longTermBondSpec(beta,theta,gamma,d0,d1,R1,1.0,coupon,assetBounds,assetN,YParams,mParams,arellanoPenalty,0.0,0.0);
#redo with ten-year bonds
LTBSpec2=longTermBondSpec(beta,theta,gamma,d0,d1,R1,Lambda/2.0,coupon,assetBounds,assetN,YParams,mParams,arellanoPenalty,smoothQ,smoothV);


#create environment:
LTBModel=makeEval(LTBSpec);
STBModel=makeEval(STBSpec);
LTBModel2=makeEval(LTBSpec2);


#solve Benchmark model:
@info "Solving Benchmark Model"
GC.gc()
@time vfiGOneStep!(LTBSpec,LTBModel,1e-10,2000,false);


#solve ST Bond model:
@info "Solving ST Bond Model"
GC.gc()
@time vfiGOneStep!(STBSpec,STBModel,1e-10,2000,false);

#solve Ten-year Bond model:
@info "Solving Ten-year Bond Model"
GC.gc()
@time vfiGOneStep!(LTBSpec2,LTBModel2,1e-10,2000,false);



############################
#Figures
############################

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


midY=UInt32(ceil(yN/2))
qstar=(Lambda+(1-Lambda)*coupon)/(R1-(1-Lambda));
qstar2=(Lambda/2+(1-Lambda/2)*coupon)/(R1-(1-Lambda/2));

#set linewidth for plots:
lw=2


#plot prices at lowest Y
f=plot(-LTBModel.aGrid,LTBModel.qGrid[:,1]/qstar,line=(lw, :black), legend=false,xlabel=(L"$b'$"),ylabel=(L"$q(b',y=y_{min})/q^{RF}$"))
plot!(f,-STBModel.aGrid,STBModel.qGrid[:,1]*R1,line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=2)
plot!(f,-LTBModel2.aGrid,LTBModel2.qGrid[:,1]/qstar2,line=(lw,:dashdot, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=2)
plot!(f,xlims=(0, -assetGrid[1]))
savefig(f, joinpath(".","Chapter7","output","fig_7_6a.pdf"))


#plot prices at mean Y
f=plot(-LTBModel.aGrid,LTBModel.qGrid[:,midY]/qstar,line=(lw, :black), legend=false,xlabel=(L"$b'$"),ylabel=(L"$q(b',y=\mu)/q^{RF}$"))
plot!(f,-STBModel.aGrid,STBModel.qGrid[:,midY],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=2)
plot!(f,-LTBModel2.aGrid,LTBModel2.qGrid[:,midY]/qstar2,line=(lw,:dashdot, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=2)
plot!(f,xlims=(0, -assetGrid[1]))
savefig(f, joinpath(".","Chapter7","output","fig_7_6b.pdf"))

#plot prices at highest Y
f=plot(-LTBModel.aGrid,LTBModel.qGrid[:,yN]/qstar,line=(lw, :black), legend=false,xlabel=(L"$b'$"),ylabel=(L"$q(b',y=y_{max})/q^{RF}$"))
plot!(f,-STBModel.aGrid,STBModel.qGrid[:,yN],line=(lw,:dash, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :circle, markersize=2)
plot!(f,-LTBModel2.aGrid,LTBModel2.qGrid[:,yN]/qstar2,line=(lw,:dashdot, :black),st = :samplemarkers, step = 20, markercolor=:black, shape = :diamond, markersize=2)
plot!(f,xlims=(0, -assetGrid[1]))
savefig(f, joinpath(".","Chapter7","output","fig_7_6c.pdf"))

############################
#Simulations
############################

#Simulation Specs
bigN=1000
bigT=20000

#Simulate Models
@time shocks=makeShockSequences(LTBSpec,LTBModel,bigT,bigN);
@info "Simulating Benchmark Model"
@time paths,moments=simulatePathsMC(LTBSpec,LTBModel,shocks,bigN,1000,20,4.0);
@info "Simulating ST Bond Model"
@time STpaths,STmoments=simulatePathsMC(STBSpec,STBModel,shocks,bigN,1000,20,4.0);
@info "Simulating Ten-year Bond Model"
@time paths2,moments2=simulatePathsMC(LTBSpec2,LTBModel2,shocks,bigN,1000,20,4.0);


rowlabels=["b'/y","mv/y","Def Rate","meanSpread","StDevSpread","StdDev(c)/StdDev(y)","corr(tb,y)","corr(spread,y)","corr(b'/y,spread)","corr(y-c,spread)"]
header=["Moment", "ST Bond Model","Benchmark LT Bond Model", "Ten-year Bond Model"]

println("#Table 7.1")
open(joinpath(".","Chapter7","output","Table7_1.txt"), "w") do f
    pretty_table(f,hcat(rowlabels,STmoments,moments,moments2),header, formatters=ft_printf("%5.3f",2:4))
end

