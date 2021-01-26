using Random

# MCPaths is a parametrically typed immutable that contains a collection of "paths" for macro aggregates indexed by time.  

struct MCPaths{F<:Real,S<:Integer}
    cSim::Array{F,2}
    ySim::Array{F,2}
    aSim::Array{F,2}
    apSim::Array{F,2}
    tbSim::Array{F,2}
    rSim::Array{F,2}
    mvSim::Array{F,2}
    defSim::Array{Bool,2}
    inDefSim::Array{Bool,2}
    yIndSim::Array{S,2}
    aIndSim::Array{S,2}
    apIndSim::Array{S,2}
    noDefDuration::Array{S,2}
    inSample::Array{Bool,2}
    inDefSample::Array{Bool,2}
end

# shockSequence is a parametrically typed immutable that contains paths of endowment shocks and re-entry shocks.  These are the model's exogenous shocks processes
struct shockSequence{F<:Real,S<:Integer}
    yIndSim::Array{S,2}
    reentryShockSim::Array{F,2}
end


#makeShockSequences simulates the exogenous shock sequences.  It takes as arguments a model specification m, a solved model s, a time horizon T for each simulated path, and the number of paths N 
function makeShockSequences(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S},bigT::S,bigN::S) where{F<:Real,S<:Integer}


    yIndSim=zeros(S,bigT,bigN)

    stDist=genStIncDist(m,s,1e-10,1000)

    stDistCDF=cumsum(stDist)

    yTDistCDF=cumsum(s.yTMat,dims=1)

    #set seed for income shocks:
    Random.seed!(1234)
    yrand=rand(bigT,bigN)

    for littleN in 1:bigN

        yIndSim[1,littleN]=searchsortedfirst(stDistCDF,yrand[1,littleN])
        for littleT in 2:bigT
            yIndSim[littleT,littleN]=searchsortedfirst(yTDistCDF[:,yIndSim[littleT-1,littleN]],yrand[littleT,littleN])
        end
    end
    #set seed for reentry shocks:
    Random.seed!(666)
    reentryShockSim=rand(bigT,bigN)

    return shockSequence(yIndSim,reentryShockSim)

end


#compute stationary distribution of income from model parameters
function genStIncDist(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S},tol::F,maxIter::S) where{F<:Real,S<:Integer}

    #initialize with uniform
    oldDist=ones(m.yParams.yPoints)/m.yParams.yPoints
    newDist=zeros(m.yParams.yPoints)

    iCount=1
    dDist=1.0+tol

    while (iCount<=maxIter)&&(dDist>tol)
        mul!(newDist,s.yTMat,oldDist)

        dDist=maximum(abs.(newDist-oldDist))

        copyto!(oldDist,newDist)
        iCount+=1
    end
    deflFac=sum(newDist)^(-1)
    return newDist*deflFac
end



# simulates path given model parameters m, solved model s, exogenous shocks ymShocks, number of simulations bigN, how many periods to drop after a default "trim" and "trimDef", and periods per year.  
function simulatePathsMC(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S},ymShocks::shockSequence{F,S},bigN::S,trim::S,trimDef::S,periodsPerYear::F) where{F<:Real,S<:Integer}
    bigT,bigNMax=size(ymShocks.yIndSim)
    @assert bigN<=bigNMax

    cSim=zeros(bigT,bigN)
    rSim=zeros(bigT,bigN)
    tbSim=zeros(bigT,bigN)
    mvSim=zeros(bigT,bigN)
    yIndSim=zeros(S,bigT,bigN)
    yIndSim.=ymShocks.yIndSim[:,1:bigN]
    reentryShocks=zeros(F,bigT,bigN)
    reentryShocks=ymShocks.reentryShockSim[:,1:bigN]
    ySim=zeros(bigT,bigN)

    aIndSim=zeros(S,bigT,bigN)
    aSim=zeros(bigT,bigN)
    apSim=zeros(bigT,bigN)
    apIndSim=zeros(S,bigT,bigN)

    defSim=zeros(Bool,bigT,bigN)
    inDefSim=zeros(Bool,bigT,bigN)
    noDefDuration=zeros(S,bigT,bigN)
    inSample=zeros(Bool,bigT,bigN)
    inDefSample=zeros(Bool,bigT,bigN)

    aIndSim[1,:].=s.a0Ind
    aSim[1,:].=0.0

    for littleN in 1:bigN

        for littleT in 1:bigT


            if inDefSim[max(1,littleT-1),littleN]==true
                reenterRand=reentryShocks[littleT,littleN]
                if reenterRand<m.theta
                    inDefSim[littleT,littleN]=false
                    noDefDuration[littleT,littleN]=1
                    aIndSim[littleT,littleN]=s.a0Ind
                else
                    inDefSim[littleT,littleN]=true
                    noDefDuration[littleT,littleN]=0
                end
            end


            if inDefSim[littleT,littleN]==true
                ySim[littleT,littleN]=s.yDefGrid[yIndSim[littleT,littleN]]
                cSim[littleT,littleN]=ySim[littleT,littleN]
            elseif s.pol.defaultGrid[aIndSim[littleT,littleN],yIndSim[littleT,littleN]]==true
                defSim[littleT,littleN]=true
                inDefSim[littleT,littleN]=true
                noDefDuration[littleT,littleN]=0
                ySim[littleT,littleN]=s.yDefGrid[yIndSim[littleT,littleN]]
                cSim[littleT,littleN]=ySim[littleT,littleN]

            else
                defSim[littleT,littleN]=false
                inDefSim[littleT,littleN]=false
                if littleT!=1
                    noDefDuration[littleT,littleN]=noDefDuration[littleT-1,littleN]+1
                end
                ySim[littleT,littleN]=s.yGrid[yIndSim[littleT,littleN]]


                apIndSim[littleT,littleN]=s.pol.apRGrid[aIndSim[littleT,littleN],yIndSim[littleT,littleN]]
                apSim[littleT,littleN]=s.aGrid[apIndSim[littleT,littleN]]

                cSim[littleT,littleN]=s.cashInHandsGrid[aIndSim[littleT,littleN],yIndSim[littleT,littleN]]-s.apRev[apIndSim[littleT,littleN],yIndSim[littleT,littleN]]
                tbSim[littleT,littleN]=ySim[littleT,littleN]-cSim[littleT,littleN]
                rSim[littleT,littleN]=s.qGrid[apIndSim[littleT,littleN],yIndSim[littleT,littleN]]^(-1)-1.0
                mvSim[littleT,littleN]=s.qGrid[apIndSim[littleT,littleN],yIndSim[littleT,littleN]]*apSim[littleT,littleN]
                if littleT<bigT
                    aIndSim[littleT+1,littleN]=apIndSim[littleT,littleN]
                    aSim[littleT+1,littleN]=apSim[littleT,littleN]
                end
            end
        end
        for littleT in (trim+1):bigT
            if (noDefDuration[littleT,littleN]>trimDef)
                inSample[littleT,littleN]=true
            end
            if (noDefDuration[littleT-1,littleN]>(trimDef-1))
                inDefSample[littleT,littleN]=true
            end
        end
    end

    meanAPFVDivY=mean(-apSim[inSample]./ySim[inSample])
    meanAPMVDivY=mean(-mvSim[inSample]./ySim[inSample])
    meanAFVNDDivY=mean(-aSim[inSample]./ySim[inSample])
    meanAFVDivY=mean(-aSim[inDefSample]./ySim[inDefSample])

    defRate=1-(1-mean(defSim[inDefSample]))^4
    meanSpread=mean((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear)
    volSpread=sqrt(var((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear))
    volCDivVolY=sqrt(var(log.(cSim[inSample]))/var(log.(ySim[inSample])))
    volTB=sqrt(var(tbSim[inSample]./ySim[inSample]))
    corTBLogY=cor(tbSim[inSample]./ySim[inSample],log.(ySim[inSample]))

    corSpreadLogY=cor((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear,log.(ySim[inSample]))
    corSpreadAPDivYM=cor((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear,-apSim[inSample]./ySim[inSample])
    corSpreadTB=cor((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear,tbSim[inSample]./ySim[inSample])

    momentVals=[meanAPFVDivY,meanAPMVDivY,defRate,meanSpread,volSpread,volCDivVolY,corTBLogY,corSpreadLogY,corSpreadAPDivYM,corSpreadTB]
    MCPathsOut=MCPaths(cSim,ySim,aSim,apSim,tbSim,rSim,mvSim,defSim,inDefSim,yIndSim,aIndSim,apIndSim,noDefDuration,inSample,inDefSample)

    #Note that assets have a negative sign in moments to map to debt

    return MCPathsOut,momentVals

end




