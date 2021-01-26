using Random
struct MCPaths{F<:Real,S<:Integer}
    cSim::Array{F,2}
    ySim::Array{F,2}
    mSim::Array{F,2}
    ymSim::Array{F,2}
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


struct shockSequence{F<:Real,S<:Integer}
    yIndSim::Array{S,2}
    mSim::Array{F,2}
    reentryShockSim::Array{F,2}
end



function makeShockSequences(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},bigT::S,bigN::S) where{F<:Real,S<:Integer,T}


    yIndSim=zeros(S,bigT,bigN)
    mSim=zeros(bigT,bigN)

    stDist=genStIncDist(m,s,1e-10,1000)

    stDistCDF=cumsum(stDist)

    mDistTN=Distributions.TruncatedNormal(m.mParams.mu,sqrt(m.mParams.epsilon2),s.income.mBounds[1],s.income.mBounds[2])
    yTDistCDF=cumsum(s.income.yTMat,dims=1)

    #set seed for income shocks:
    Random.seed!(1234)
    yrand=rand(bigT,bigN)

    for littleN in 1:bigN

        yIndSim[1,littleN]=searchsortedfirst(stDistCDF,yrand[1,littleN])
        mSim[1,littleN]=rand(mDistTN)
        for littleT in 2:bigT
            yIndSim[littleT,littleN]=searchsortedfirst(yTDistCDF[:,yIndSim[littleT-1,littleN]],yrand[littleT,littleN])
            mSim[littleT,littleN]=rand(mDistTN)
        end
    end
    reentryShockSim=rand(bigT,bigN)

    return shockSequence(yIndSim,mSim,reentryShockSim)

end


# function writeYMShocksBin(ymShocks::shockSequence{F,S},filePrefix::String,fileDir::String) where{F<:Real,S<:Integer}

#     write(joinpath(fileDir,filePrefix*"_yIndSim.bin"),ymShocks.yIndSim)
#     write(joinpath(fileDir,filePrefix*"_mSim.bin"),ymShocks.mSim)
#     write(joinpath(fileDir,filePrefix*"_reentryShockSim.bin"),ymShocks.reentryShockSim)

# end

# function readYMShocksBin(m::longTermBondSpec{F,S},filePrefix::String,fileDir::String,numCols::S) where{F<:Real,S<:Integer}

#     yIndSimB=read(joinpath(fileDir,filePrefix*"_yIndSim.bin"))
#     mSimB=read(joinpath(fileDir,filePrefix*"_mSim.bin"))
#     reentryShockSimB=read(joinpath(fileDir,filePrefix*"_reentryShockSim.bin"))



# 	yIndSimLong=reinterpret(S,yIndSimB)
# 	mSimLong=reinterpret(F,mSimB)
#     reentryShockSimLong=reinterpret(F,reentryShockSimB)

# 	numRows=div(length(yIndSimLong),numCols)

# 	yIndSim=zeros(S,numRows,numCols)
# 	yIndSim[:].=yIndSimLong
# 	mSim=zeros(F,numRows,numCols)
# 	mSim[:].=mSimLong
#     reentryShockSim=zeros(F,numRows,numCols)
#     reentryShockSim[:].=reentryShockSimLong

#     return shockSequence(yIndSim,mSim,reentryShockSim)

# end

#compute stationary distribution of income
function genStIncDist(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},tol::F,maxIter::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}

    oldDist=ones(m.yParams.yPoints)/m.yParams.yPoints
    newDist=zeros(m.yParams.yPoints)

    iCount=1
    dDist=1.0+tol

    while (iCount<=maxIter)&&(dDist>tol)
        mul!(newDist,s.income.yTMat,oldDist)

        dDist=maximum(abs.(newDist-oldDist))

        copyto!(oldDist,newDist)
        iCount+=1
    end
    deflFac=sum(newDist)^(-1)
    return newDist*deflFac
end




function simulatePathsMC(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},ymShocks::shockSequence{F,S},bigN::S,trim::S,trimDef::S,periodsPerYear::F) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
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
    mSim=zeros(bigT,bigN)
    mSim.=ymShocks.mSim[:,1:bigN]
    ymSim=zeros(bigT,bigN)
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
                ySim[littleT,littleN]=s.income.yDefGrid[yIndSim[littleT,littleN]]
                ymSim[littleT,littleN]=ySim[littleT,littleN]+mSim[littleT,littleN]
                cSim[littleT,littleN]=ymSim[littleT,littleN]
            elseif (s.pol.alwaysDefault[aIndSim[littleT,littleN],yIndSim[littleT,littleN]]==true)||((s.pol.neverDefault[aIndSim[littleT,littleN],yIndSim[littleT,littleN]]==false)&&(mSim[littleT,littleN]<s.pol.defThreshold[aIndSim[littleT,littleN],yIndSim[littleT,littleN]]))
                defSim[littleT,littleN]=true
                inDefSim[littleT,littleN]=true
                noDefDuration[littleT,littleN]=0
                ySim[littleT,littleN]=s.income.yDefGrid[yIndSim[littleT,littleN]]
                ymSim[littleT,littleN]=ySim[littleT,littleN]+s.income.mBounds[1]
                cSim[littleT,littleN]=ymSim[littleT,littleN]

            else
                defSim[littleT,littleN]=false
                inDefSim[littleT,littleN]=false
                if littleT!=1
                    noDefDuration[littleT,littleN]=noDefDuration[littleT-1,littleN]+1
                end
                ySim[littleT,littleN]=s.income.yGrid[yIndSim[littleT,littleN]]
                ymSim[littleT,littleN]=ySim[littleT,littleN]+mSim[littleT,littleN]
                apPolLength=s.pol.mListLength[aIndSim[littleT,littleN],yIndSim[littleT,littleN]]
                aPolicyInd=searchsortedfirst(s.pol.mAPThreshold[yIndSim[littleT,littleN]][1:apPolLength,aIndSim[littleT,littleN]],mSim[littleT,littleN])
                apIndSim[littleT,littleN]=s.pol.apPolicy[yIndSim[littleT,littleN]][aPolicyInd,aIndSim[littleT,littleN]]
                apSim[littleT,littleN]=s.aGrid[apIndSim[littleT,littleN]]

                cSim[littleT,littleN]=s.netRevM0A0Grid[aIndSim[littleT,littleN],yIndSim[littleT,littleN]]-s.qGrid[apIndSim[littleT,littleN],yIndSim[littleT,littleN]]*s.aGridIncr[apIndSim[littleT,littleN],aIndSim[littleT,littleN]]+mSim[littleT,littleN]
                tbSim[littleT,littleN]=ymSim[littleT,littleN]-cSim[littleT,littleN]
                rSim[littleT,littleN]=(m.lambda+(1.0-m.lambda)*m.coup)/s.qGrid[apIndSim[littleT,littleN],yIndSim[littleT,littleN]]-m.lambda
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

    meanAPFVDivY=mean(-apSim[inSample]./ymSim[inSample])
    meanAPMVDivY=mean(-mvSim[inSample]./ymSim[inSample])
    meanAFVNDDivY=mean(-aSim[inSample]./ymSim[inSample])
    meanAFVDivY=mean(-aSim[inDefSample]./ymSim[inDefSample])

    #defRate=mean(defSim[inDefSample])
    defRate=1-(1-mean(defSim[inDefSample]))^4
    meanSpread=mean((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear)
    volSpread=sqrt(var((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear))
    volCDivVolY=sqrt(var(log.(cSim[inSample]))/var(log.(ymSim[inSample])))
    volTB=sqrt(var(tbSim[inSample]./ymSim[inSample]))
    corTBLogY=cor(tbSim[inSample]./ymSim[inSample],log.(ymSim[inSample]))

    corSpreadLogY=cor((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear,log.(ymSim[inSample]))
    corSpreadAPDivYM=cor((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear,-apSim[inSample]./ymSim[inSample])
    corSpreadTB=cor((1.0 .+rSim[inSample]).^periodsPerYear .-m.R^periodsPerYear,tbSim[inSample]./ymSim[inSample])


#    momentVals=[meanAPFVDivY,meanAPMVDivY,meanAFVNDDivY,meanAFVDivY,defRate,meanSpread,volSpread,volCDivVolY,volTB,corTBLogY,corSpreadLogY,corSpreadAPDivYM,corSpreadTB,corAPDivYLogY]
    momentVals=[meanAPFVDivY,meanAPMVDivY,defRate,meanSpread,volSpread,volCDivVolY,corTBLogY,corSpreadLogY,corSpreadAPDivYM,corSpreadTB]
    MCPathsOut=MCPaths(cSim,ySim,mSim,ymSim,aSim,apSim,tbSim,rSim,mvSim,defSim,inDefSim,yIndSim,aIndSim,apIndSim,noDefDuration,inSample,inDefSample)
 
    #Note that assets have a negative sign in moments to map to debt

    return MCPathsOut,momentVals

end




