# This code was primarily written by Stelios Fourakis.   We are greatly indebted for his work on this project.  All errors are our own.


#The following library of functions provides types and methods for:

#1. defining and setting up the objects needed to solve the model described in Arellano (2008)
#2. solving the model itself, either via the outer loop-inner loop procedure described in Arellano (2008) or the one step procedure which has since become more popular
#3. calculating the stationary, ergodic distribution of default state, income, and borrowing, given a specification of the income process and government policy functions


#The code implementing Tauchen's method for approximating an AR(1) process via a Markov Chain is largely taken from https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/markov/markov_approx.jl

#We have used the following convention in for loops, especially in nested ones:
#1. i iterates over values of the income process
#2. j iterates over values of current period borrowing
#3. jj iterates over values of next period borrowing
#Except where otherwise noted, the first dimension of an array corresponds to the borrowing state and the second to the income process state

#Throughout, we use the convention that a is an asset, so negative amounts indicate net liabilities.

#The code relies largely on a set of parametrically typed immutable objects which are defined at the beginning of the code. These types are:
#1. ar1Params - includes the specification of AR(1) process as well as the range and resolution of the Markov chain to be used to approximate it
#2. deptSpecOnePeriod - includes value of basic parameters of the model (beta, R, etc.), the specification of the income process, and the grid for borrowing
#3. onePeriodDebtEval - includes the income process, value function grids, policy function grids, and some extra limits on borrowing (some arbitrary, others to speed up computation of the model)
#4. debtDist - includes the joint distribution of default, income, and borrowing


using Distributions, SpecialFunctions, LinearAlgebra

#ar1Params is a parametrically typed immutable containing:
#1. yPoints: the number of points to be used in discretizing the AR(1) Process
#2. rho: the persistence parameter of the process
#3. eta2: the variance of the innovations of the process
#4. mu: the permanent component of the process
#5. stdSpan: the number of standard deviations to each side of the mean which should fall within the grid
#Throughout, we use the convention y_t=mu+rho*y_t-1+sqrt(eta2)*e_t

struct ar1Params{F<:Real,S<:Integer}
    yPoints::S
    rho::F
    eta2::F
    mu::F
    stdSpan::F
    inflateEndpoints::Bool
end

#gridSearchParams is a parametrically typed immutable containing the index information required to quickly access point and bound indices to perform a bisection style grid search which exploits policy function monotonicity
#Its contents are:
#1. boundGrids: a one dimensional array containing the one dimensional arrays of bound indices at each level
#2. pointGrids:a one dimensional array containing the one dimensional arrays of point indices at each level
#3. pointLB: a one dimensional array containing the one dimensional arrays of lower bound indices for each point at each level
#4. pointUB: a one dimensional array containing the one dimensional arrays of upper bound indices for each point at each level
#5. boundGridPoints: a one dimensional array containing the length of the arrays in boundGrids
#6. levelPoints: a one dimensional array containing the length of the arrays in pointGrids (and therefore also in pointLB and pointUB)
#7. numLevels: the length all the above arrays (at the top level, i.e. the number of actual arrays contained in boundGrids rather than the number of points in all the arrays it contains)
struct gridSearchParams{S<:Integer}
    boundGrids::Array{Array{S,1},1}
    pointGrids::Array{Array{S,1},1}
    pointLB::Array{Array{S,1},1}
    pointUB::Array{Array{S,1},1}
    boundGridPoints::Array{S,1}
    levelPoints::Array{S,1}
    numLevels::S
end

#onePeriodDebtSpec is a parametrically typed immutable containing enough of the information required to construct the full model. Its contents are:
#1. beta: the government's discount factor
#2. theta: the probability of reentry to international markets when in financial autarky
#3. gamma: the coefficient of relative risk aversion for CRRA preferences
#4. hpen0: either the h in y_def(s)=min(y(s),h*E[y(s)]), i.e. the output penalty when in default, or d0 in y_def(s)=y(s)-max(0,d0*y(s)+d1*y(s)^2)
#5. hpen1: d1 in y_def(s)=y(s)-max(0,d0*y(s)+d1*y(s)^2)
#6. R: the gross interational interest rate
#7. yParams: the specification of the income process approximation
#8. aGrid: a one dimensional array of asset values
#10. simplePen: a boolean variable which is true if y_def(s)=min(y(s),h*E[y(s)]) and false if y_def(s)=y(s)-max(0,d0*y(s)+d1*y(s)^2)

#Throughout the code, the variable m is always an object of type onePeriodDebtSpec

struct onePeriodDebtSpec{F<:Real,S<:Integer}
    beta::F
    theta::F
    gamma::F
    hpen0::F
    hpen1::F
    R::F
    yParams::ar1Params{F,S}
    aGrid::Array{F,1}
    simplePen::Bool
end

#vFunc is a parametrically typed immutable containing the flow utilities while in default and
#the value function/continuation value function grids. Its contents are:
#1. vDFlow: a one dimensional array (axes are 1. current income) containing the the value of flow utility conditional on being in default
#2. vGrid: a two dimensional array (axes are 1. current asset level and 2. current income) containing the beginning of period value function (taking default decisions as given)
#3. EVGrid: a two dimensional array (axes are 1. next period asset level and 2. current persistent income) containing
#the expected continuation value function Z(y,a')=E[V(y',a')|y]
#4. vDGrid: a one dimensional array (axes are 1. current income) containing the value to the government under default
#5. EVDGrid: a one dimensional array (axes are 1. current income) containing
#the grid of the expected continuation value function Z^D(y)=E[theta*V(y',0)+(1.0-theta)*V^D(y')|y]

struct vFunc{F<:Real}
    vDFlow::Array{F,1}
    vGrid::Array{F,2}
    vDGrid::Array{F,1}
    EVGrid::Array{F,2}
    EVDGrid::Array{F,1}
end

#policies is a parametrically typed immutable containing the government's policy functions. Its contents are:
#1. apRGrid: a two dimensional array (axes are 1. current asset level and 2. current income) containing the index of the government's asset choice in the asset grid
#2. defaultGrid: a two dimensional array (axes are 1. current asset level and 2. current income) containing a true/false marker for whether the government defaults.

struct policies{S<:Integer}
    apRGrid::Array{S,2}
    defaultGrid::Array{Bool,2}
end


#onePeriodDebtEval is a parametrically typed immutable which provides, in combination with a onePeriodDebtSpec, all the objects and information required to solve the model. Its contents are:
#1. yGrid: a one dimensional array of the values of the discretized income process
#2. yDefGrid: a one dimensional array of the values of the discretized income process while in default
#3. yTMat: a two dimensional array containing the transition probabilities for the discretized income process, using the convention that each column sums to 1, i.e. yTMat[i,j] is the probability of transitioning to state i given that the current state is j
#4. yMean: the long run mean of the income process
#5. VF: an object containing the model's value functions
#6. pol: an object containing the model's policy functions
#7. cashInHandsGrid: a two dimensional array (axes are 1. incoming borrowing, 2. income) containg the values of income plus assets
#8. qGrid: a two dimensional array (axes are 1. next period borrowing, 2. income) containing the bond price function
#9. apRev: a two dimensional array (axes are 1. next period borrowing, 2. income) containing the cost of saving each asset amount while in each income state
#10. a0Ind: the index of 0 in the current period borrowing grid
#11. maxAPInd: a one dimensional array (axes are 1. incoming borrowing, 2. income) containing the maximum positive asset position that the government could save conditional on not defaulting.
#12. apSearchParams: the set of objects required for exploiting the binary monotonicity of the default and savings policy functions when solving the government's problem
#13. noConvergeMark: a one element array to keep track of whether the value functions have converged to a specified tolerance
#14. adim: the length of the asset grid
#15. aGrid: the copy of the asset grid


#Throughout the code, the variable s is always an object of type onePeriodDebtEval


struct onePeriodDebtEval{F<:Real,S<:Integer}
    yGrid::Array{F,1}
    yDefGrid::Array{F,1}
    yTMat::Array{F,2}
    yMean::F
    VF::vFunc{F}
    pol::policies{S}
    cashInHandsGrid::Array{F,2}
    qGrid::Array{F,2}
    apRev::Array{F,2}
    a0Ind::S
    maxAPInd::Array{S,2}
    apSearchParams::gridSearchParams{S}
    noConvergeMark::Array{Bool,1}
    adim::S
    aGrid::Array{F,1}
end


#debtDist is a parametrically typed immutable describing the stationary, ergodic joint distirbution of default state, income state, and current asset state. Its contents are:
#1. repayDist: a two dimensional array (axes are 1. incoming assets, 2. income) containing the noninflated joint distribution of income and assets, conditional on the government having access to financial markets (where by noninflated, we mean that we have not divided the conditional distribution by the probability of the condition being true)
#2. defaultDist: a one dimensional array (axes are 1. income) containing the noninflated distribution of income, conditional on the government being in default
struct debtDist{F<:Real}
    repayDist::Array{F,2}
    defaultDist::Array{F,1}
end


#onePeriodDebtUpdate is a parametrically typed immutable which contains all the objects and information required to update a guess for the model. Its contents are:
#1. VF: an object containing the model's value functions
#2. pol: an object containing the model's policy functions
#3. qGrid: a two dimensional array (axes are 1. next period assets, 2. income) containing the bond price function
#4. solveMark: a two dimensional array (axes are 1. incoming assets, 2. income) which is updated during each iteration to mark at which points the government's problem under repayment was actually solved
#5. feasibleSolution: a two dimensional array (axes are 1. incoming assets, 2. income) which is updated during each iteration to mark at which points the government's problem under repayment was actually solved and where the solution resulted in strictly positive consumption
#6. maxADefInd: a one dimensional array (axes are 1. income) containing the index of the highest asset level at which default occurs in each income state
#7. EVA0: a one dimensional array (axes are 1. income) containing the continuation value of reentry

#Throughout the code, the variable g2 is always an object of type onePeriodDebtUpdate

struct onePeriodDebtUpdate{F<:Real,S<:Integer}
    VF::vFunc{F}
    pol::policies{S}
    qGrid::Array{F,2}
    solveMark::Array{Bool,2}
    feasibleSolution::Array{Bool,2}
    maxADefInd::Array{S,1}
    EVA0::Array{F,1}
end



#Various routines to discretize AR(1) processes
#@author : Spencer Lyon <spencer.lyon@nyu.edu>
#@date : 2014-04-10 23:55:05
#References
#----------
#https://lectures.quantecon.org/jl/finite_markov.html


#Define versions of the standard normal cdf for use in the tauchen method
std_norm_cdf(x::T) where {T <: Real} = 0.5 * erfc(-x/sqrt(2))
std_norm_cdf(x::Array{T}) where {T <: Real} = 0.5 .* erfc(-x./sqrt(2))


# #The function tauchen(g) generates a discretized version of an AR(1) process using the convention:
# #y_t=mu+rho*y_t-1+sqrt(eta2)*e_t
# #Its only argument is a variable g of type ar1Params.
# #Its output is a tuple of 3 objects. They are:
# #1. yOut: a one dimensional array of values of the discretized process
# #2. yTMatOut: a two dimensional array containing transition probabilities for the process, using the convention that yTMatOut[i,j] is the probability of transitioning to state i conditional on being in state j
# #3. yMeanOut: the analytical long run mean of the process
# Modified to allow option of whether truncated mass is placed on endpoints 
#(inflateEndpoints==true) or placed uniformly over grid (inflateEndpoints==false)
function tauchenRev(g::ar1Params{F,S}) where{F<:Real,S<:Integer}
    # Get discretized space
    eta=sqrt(g.eta2)
    a_bar = g.stdSpan * sqrt(g.eta2 / (1 - g.rho^2))
    y = LinRange(-a_bar, a_bar, g.yPoints)
    d = y[2] - y[1]

    # Get transition probabilities
    yTMat = zeros(g.yPoints, g.yPoints)
    if g.inflateEndpoints==true
        for i = 1:g.yPoints
            # Do end points first
            yTMat[1,i] = std_norm_cdf((y[1] - g.rho*y[i] + d/2) / sqrt(g.eta2))
            yTMat[g.yPoints,i] = 1 - std_norm_cdf((y[g.yPoints] - g.rho*y[i] - d/2) / sqrt(g.eta2))

            # fill in the middle columns
            for j = 2:g.yPoints-1
                yTMat[j, i] = (std_norm_cdf((y[j] - g.rho*y[i] + d/2) / sqrt(g.eta2)) -
                               std_norm_cdf((y[j] - g.rho*y[i] - d/2) / sqrt(g.eta2)))
            end
        end
    else

        for i = 1:g.yPoints
            for j = 1:(g.yPoints)
                yTMat[j, i] = (std_norm_cdf((y[j] - g.rho*y[i] + d/2) / sqrt(g.eta2)) -
                               std_norm_cdf((y[j] - g.rho*y[i] - d/2) / sqrt(g.eta2)))
            end
        end

    end

    # NOTE: We need to shift this vector after finding probabilities
    #       because when finding the probabilities we use a function
    #       std_norm_cdf that assumes its input argument is distributed
    #       N(0, 1). After adding the mean E[y] is no longer 0.
    #

    yOut = exp.(y .+ g.mu / (1 - g.rho))

    yTMatSums=sum(yTMat,dims=1)
    for i in 1:g.yPoints
        yTMat[:,i].*=yTMatSums[i]^-1
    end

    #Since, in the long run, ln(y_t) is distributed normally with mean m=g.mu / (1 - g.rho) and variance s2=g.eta2/(1-g.rho^2),
    #y_t is distributed lognormally in the long run, and the mean of such a lognormal variable is exp(m+1/2*s2)
    yMeanOut=exp(g.mu / (1 - g.rho)+1/2*g.eta2/(1-g.rho^2))

    return yOut, yTMat, yMeanOut
end



#genBisectSearchGrids is a utility function for constructing a gridSearchParams object
#Its only argument is the number of points in the grid of interest
#Its output is simply an object of type gridSearchParams
function genBisectSearchGrids(numPoints::S) where{S<:Integer}
    #Initialize all the components of the output object
    numLevels=S(ceil(log(numPoints-1)/log(2)))
    levelPoints=zeros(S,numLevels)
    boundGridPoints=zeros(S,numLevels)

    boundGrids=Array{Array{S,1},1}(undef,numLevels)
    pointGrids=Array{Array{S,1},1}(undef,numLevels)
    pointLB=Array{Array{S,1},1}(undef,numLevels)
    pointUB=Array{Array{S,1},1}(undef,numLevels)

    #Initialize a uncleaned version of boundGrids
    boundGridsFull=Array{Array{S,1},1}(undef,numLevels)
    boundGridsFull[1]=[1,numPoints]

    #Set the values of each objects at the first level
    boundGrids[1]=[1,numPoints]
    pointGrids[1]=[S(ceil(numPoints/2))]
    levelPoints[1]=1
    boundGridPoints[1]=2
    pointLB[1]=[1]
    pointUB[1]=[numPoints]

    #Iterate over levels
    for j in 2:numLevels
        #Set the current number of points to 0
        levelPoints[j]=0

        #Construct the set of points available to be used as upper or lower bounds
        tempBoundGrid=vcat(boundGridsFull[j-1],pointGrids[j-1])

        #Sort this array and remove all duplicates
        sort!(tempBoundGrid,alg=QuickSort)
        notFinishedCleaning=true
        tempInd=2
        while notFinishedCleaning==true
            if tempBoundGrid[tempInd]==tempBoundGrid[tempInd-1]
                deleteat!(tempBoundGrid,tempInd)
                tempInd==2
            else
                tempInd+=1
            end
            if tempInd>length(tempBoundGrid)
                notFinishedCleaning=false
            end
        end
        #Record the current list of available bounds
        boundGridsFull[j]=deepcopy(tempBoundGrid)
        #Initialize a set of arrays which will mark whether or not the point in the temporary bound grid is actually needed to calculate any of the new points
        tempKeepMark=ones(Bool,length(tempBoundGrid))
        tempKeepMark[1]=ifelse((tempBoundGrid[1]+1)==tempBoundGrid[2],false,true)
        tempKeepMark[end]=ifelse((tempBoundGrid[end-1]+1)==tempBoundGrid[end],false,true)
        #If necessary, iterate over the elements of the temporary bound grid
        #Check whether each interior element is both 1 greater than the previous element and 1 less than the next element. If this is true, the point is not a bound index for any points at this level
        #Also check whether each element except the first is equal to the element preceding it. Whenever this is the case, a new point will be generated, and that must be reflected by levelPoints

        if length(tempBoundGrid)>2
            for i in 2:(length(tempBoundGrid)-1)
                if ((tempBoundGrid[i-1])==(tempBoundGrid[i]-1))&&((tempBoundGrid[i+1])==(tempBoundGrid[i]+1))
                    tempKeepMark[i]=false
                end
            end
            for i in 2:(length(tempBoundGrid))
                if ((tempBoundGrid[i-1])!=(tempBoundGrid[i]-1))
                    levelPoints[j]+=1
                end
            end
        end
        #Generate the the current boundGrid, record its length, and initialize the grid of point indices and lower and upper bound indices at this level
        boundGrids[j]=tempBoundGrid[tempKeepMark]
        boundGridPoints[j]=length(boundGrids[j])
        pointGrids[j]=zeros(S,levelPoints[j])
        pointLB[j]=zeros(S,levelPoints[j])
        pointUB[j]=zeros(S,levelPoints[j])

        #Initialize a variable tracking the index in pointGrids[j] next point to be added
        tempInd=1
        #Alternate between using ceilings and floors
        if isodd(j)==true
            #Iterate over the elements of the temporary bound grid
            for k in 1:(length(tempBoundGrid)-1)
                #Check whether a point needs to be added and, if necessary, add it and record the relevant upper and lower bound indices
                if tempBoundGrid[k]!=(tempBoundGrid[k+1]-1)
                    pointGrids[j][tempInd]=S(ceil((tempBoundGrid[k]+tempBoundGrid[k+1])/2))
                    pointLB[j][tempInd]=tempBoundGrid[k]
                    pointUB[j][tempInd]=tempBoundGrid[k+1]

                    tempInd+=1
                end
            end
        else
            #Iterate over the elements of the temporary bound grid
            for k in 1:(length(tempBoundGrid)-1)
                #Check whether a point needs to be added and, if necessary, add it and record the relevant upper and lower bound indices
                if tempBoundGrid[k]!=(tempBoundGrid[k+1]-1)
                    pointGrids[j][tempInd]=S(floor((tempBoundGrid[k]+tempBoundGrid[k+1])/2))
                    pointLB[j][tempInd]=tempBoundGrid[k]
                    pointUB[j][tempInd]=tempBoundGrid[k+1]

                    tempInd+=1
                end
            end
        end
    end
    return gridSearchParams(boundGrids,pointGrids,pointLB,pointUB,boundGridPoints,levelPoints,numLevels)
end



#The function makeValueFunction constructs a value function object based on the contents of a model specification. Its only argument is:
#1. m: a model specification
#Its output is simply a vFunc object
function makeValueFunction(m::onePeriodDebtSpec{F,S}) where{F<:Real,S<:Integer}
    #Set some aliases
    ydim=m.yParams.yPoints
    adim=length(m.aGrid)
    #Initialize the value function arrays
    vGrid=zeros(adim,ydim)
    vDGrid=zeros(ydim)
    EVGrid=zeros(adim,ydim)
    EVDGrid=zeros(ydim)

    #Initialize the grid of flow utility values
    vDFlow=zeros(ydim)


    return vFunc(vDFlow,vGrid,vDGrid,EVGrid,EVDGrid)
end

#The function makePolicyGrids constructs a policy function object based on the contents of a model specification.
#Its only argument is:
#1. m: a model specification
#Its output is simply a vFunc object

function makePolicyGrids(m::onePeriodDebtSpec{F,S}) where{F<:Real,S<:Integer}
    #Set some aliases
    ydim=m.yParams.yPoints
    adim=length(m.aGrid)
    #Initialize the collection of objects which define the government's policy functions
    apRGrid=ones(S,adim,ydim)
    defaultGrid=zeros(Bool,adim,ydim)


    return policies(apRGrid,defaultGrid)
end



#u is the CRRA utility function. Its arguments are just:
#1. m: the model specification
#2. x: the value of consumption
function u(m::onePeriodDebtSpec{F,S},x::U) where{F<:Real,U<:Real,S<:Integer}
    #If x is positive return the value of CRRA utility
    if x>0.0
        if m.gamma!=one(F)
            return x^(1.0-m.gamma)/(1.0-m.gamma)
        else
            return log(x)
        end
    #Otherwise return either a very negative value, scaled by how negative x is when utility is negative near x=0
    #or a large negative value which gets smaller as x rises to 0.
    #This is done to ensure that any optimization algorithm knows to try to make consumption positive at essentially any cost.
    else
        if m.gamma>=one(F)
            return u(m,1e-10)*(1.0+abs(x))
        else
            return -1e10*abs(x)
        end
    end
end


#The function makeEval constructs a onePeriodDebtEval object based on the contents of a model specification and possibly a initial bond price. Its only argument is:
#1. m: a model specification
#Its output is:
#1. outputEval: a collection of objects to be used in solving the model


function makeEval(m::onePeriodDebtSpec{F,S}) where{F<:Real,S<:Integer}
    ydim=m.yParams.yPoints
    adim=length(m.aGrid)

    #Generate the grid of income values, the transition matrix, and the long run mean of the process
    yGrid,yTMat,yMean=tauchenRev(m.yParams)


    yDefGrid=zeros(ydim)
    for i in 1:ydim
        if m.simplePen==true
            yDefGrid[i]=min(yGrid[i],m.hpen0*yMean)
        else
            yDefGrid[i]=yGrid[i]*(1.0-max(0,m.hpen0+m.hpen1*yGrid[i]))
        end
    end
    @assert minimum(yDefGrid)>0.0
    @assert minimum(yDefGrid[2:end].-yDefGrid[1:end-1])>=0

    cashInHandsGrid=zeros(adim,ydim)

    for i in 1:ydim
        for j in 1:adim
            cashInHandsGrid[j,i]=yGrid[i]+m.aGrid[j]
        end
    end


    qGrid=ones(adim,ydim)*m.R^(-1)

    apRev=zeros(adim,ydim)

    for i in 1:ydim
        for jj in 1:adim
            apRev[jj,i]=m.aGrid[jj]*m.R^(-1)
        end
    end

    #Initialize the arrays of nonbinding borrowing limits
    maxAPInd=ones(S,adim,ydim)


    #Find the location of 0.0 in the asset grid
    a0Ind=searchsortedfirst(m.aGrid,zero(F))

    #Check that the result is actually 0.0, i.e. the grid has not been misspecified as an evenly spaced grid which skips 0.0
    @assert m.aGrid[a0Ind]==zero(F)

    #Fill the array of maximum asset value indices

    #iterate over income values
    for i in 1:ydim
        #iterate over current period borrowing levels
        for j in 1:adim
            tempMaxAPInd=adim
            maxAPInd[j,i]=a0Ind
            while tempMaxAPInd>a0Ind
                if (yGrid[i]+m.aGrid[j]-m.aGrid[tempMaxAPInd]*m.R^(-1))>0
                    maxAPInd[j,i]=tempMaxAPInd
                    break
                else
                    tempMaxAPInd+=-1
                end
            end
        end
    end



    #Generate the grid search parameters and the arrays used to track their applicability
    apSearchParams=genBisectSearchGrids(adim)

    #Initialize the value functions and policies

    VF=makeValueFunction(m)
    pol=makePolicyGrids(m)

    #Fill the array of flow utilities under default
    for i in 1:ydim
        VF.vDFlow[i]=u(m,yDefGrid[i])
    end

    aGrid=deepcopy(m.aGrid)

    #construct and return the onePeriodDebtEval object whose components are constructed above
    outputEval=onePeriodDebtEval(yGrid,yDefGrid,yTMat,yMean,VF,pol,cashInHandsGrid,qGrid,apRev,a0Ind,maxAPInd,apSearchParams,[true],adim,aGrid)


    return outputEval

end

function makeUpdate(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S}) where{F<:Real,S<:Integer}
    ydim=m.yParams.yPoints
    adim=s.adim
    VF=deepcopy(s.VF)
    pol=deepcopy(s.pol)
    qGrid=deepcopy(s.qGrid)
    solveMark=zeros(Bool,adim,ydim)
    feasibleSolution=zeros(Bool,adim,ydim)

    EVA0=zeros(ydim)
    copyto!(EVA0,VF.EVGrid[s.a0Ind,:])

    maxADefInd=ones(S,ydim)

    return onePeriodDebtUpdate(VF,pol,qGrid,solveMark,feasibleSolution,maxADefInd,EVA0)
end

function updateAPRev!(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S}) where{F<:Real,S<:Integer}
    ydim=m.yParams.yPoints
    adim=s.adim
    Threads.@threads for i in 1:ydim
    #for i in 1:ydim
        for jj in 1:adim
            s.apRev[jj,i]=s.qGrid[jj,i]*s.aGrid[jj]
        end
    end

    return s
end

function Base.copyto!(dest::vFunc{F},src::vFunc{F}) where{F<:Real}
    copyto!(dest.vGrid,src.vGrid)
    copyto!(dest.EVGrid,src.EVGrid)
    copyto!(dest.vDGrid,src.vDGrid)
    copyto!(dest.EVDGrid,src.EVDGrid)

    return dest
end

function Base.copyto!(dest::policies{S},src::policies{S}) where{S<:Integer}
    copyto!(dest.apRGrid,src.apRGrid)
    copyto!(dest.defaultGrid,src.defaultGrid)

    return dest
end


#solveRepayRow! is a function which solves the government's problem for a single income state. It takes advantage of the fact that, for any single income level, default at one level of debt implies default at all higher levels of debt to speed up the computation. Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. g2: a collection of objects used to update the current guess
#4. i: the index of the income state for which the government's problem is to be solved
#5. fullAPRUpdate: a binary variable indicating whether a full update of the conditional savings policy function is required.
#Its output is a modified version of g2

function solveRepayRow!(m::onePeriodDebtSpec{F,U},s::onePeriodDebtEval{F,U},g2::onePeriodDebtUpdate{F,U},i::U,fullAPRUpdate::Bool) where{F<:Real,U<:Integer}
    #Reset the relevant column in the arrays tracking whether the repayment problem has been solved and, if so, whether that solution resulted in strictly positive consumption
    g2.solveMark[:,i].=false
    g2.feasibleSolution[:,i].=false
    g2.maxADefInd[i]=1

    tempAInd=1

    #Initialize temporary variables containing the lower and upper bounds for the index of next period borrowing
    tempLB=1

    tempUB=s.maxAPInd[1,i]

    #Solve the problem for the lowest level of net debt
    solveRepayCell!(m,s,g2,i,tempAInd,tempLB,tempUB)
    g2.solveMark[1,i]=true



    tempAInd=s.adim
    #Set the max of the minimum value which results in strictly positive consumption and the solution value as the lower bound for solving the problem at the highest level of debt
    tempUB=s.maxAPInd[s.adim,i]

    #Solve the problem for the highest level of net debt
    solveRepayCell!(m,s,g2,i,tempAInd,tempLB,tempUB)


    g2.solveMark[s.adim,i]=true


    #Iterate over the levels in the bisection grid search
    for levelInd in 1:(s.apSearchParams.numLevels)
        #Iterate over the points of the current level
        for pointInd in 1:(s.apSearchParams.levelPoints[levelInd])
            #Get the index of the current borrowing value
            tempAInd=s.apSearchParams.pointGrids[levelInd][pointInd]
            #If we have not yet solved the problem for this value during the current iteration, solve it
            if g2.solveMark[tempAInd,i]==false
                #Get the indices of the borrowing values whose policies give us an upper and lower bound on government policy at the current point
                tempUBInd=s.apSearchParams.pointUB[levelInd][pointInd]
                tempLBInd=s.apSearchParams.pointLB[levelInd][pointInd]

                #If the solution at each bound was feasible, use it (assuming that the minimum value of borrowing which results in positive consumption is not an even better lower bound)
                tempLB=ifelse(g2.feasibleSolution[tempLBInd,i]==true,g2.pol.apRGrid[tempLBInd,i],1)
                tempUB=ifelse(g2.feasibleSolution[tempUBInd,i]==true,min(g2.pol.apRGrid[tempUBInd,i],s.maxAPInd[tempAInd,i]),s.maxAPInd[tempAInd,i])

                #Solve the problem using the lower and upper bounds for next period borrowing derived above
                solveRepayCell!(m,s,g2,i,tempAInd,tempLB,tempUB)

                #Mark that we have solved the problem at this point
                g2.solveMark[tempAInd,i]=true

                #If the government defaulted at this level of debt and we do not need to solve the problem under repayment completely, do the following four things:
                #Set the default policy function for every point between the current one and the lowest value of debt at which we have thusfar observed the government default to be true
                #Update the main value function accordingly
                #Mark all of those points as having been solved
                #Update the lowest index at which we have observed the government default (tempBInd should ALWAYS be lower than tempMinDefInd, so we need not check whether that is the case)
                if (g2.pol.defaultGrid[tempAInd,i]==true)&&(fullAPRUpdate==false)
                    for tempLowerAInd in g2.maxADefInd[i]:(tempAInd-1)
                        g2.pol.defaultGrid[tempLowerAInd,i]=true
                        g2.VF.vGrid[tempLowerAInd,i]=g2.VF.vDGrid[i]

                        g2.solveMark[tempLowerAInd,i]=true
                    end

                    g2.maxADefInd[i]=max(g2.maxADefInd[i],tempAInd)
                end
            end
        end
    end

    return g2
end

#solveRepayCell! is a mutating function which solves the government's problem under repayment at exactly one point (income, current borrowing)
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. g2: a collection of objects used to update the current guess
#4. i: the index of the income state for which the government's problem is to be solved
#5. j: the index of the current borrowing state for which the government's problem is to be solved
#6. lb: the lower bound of the indices of next period borrowing for comparison
#7. ub: the upper bound of the indices of next period borrowing for comparison

#Its output is a modified version of g2

function solveRepayCell!(m::onePeriodDebtSpec{F,U},s::onePeriodDebtEval{F,U},g2::onePeriodDebtUpdate{F,U},i::U,j::U,lb::U,ub::U) where{F<:Real,U<:Integer}

    #Set the values of borrowing, the index of the borrowing value, and the value of repayment to the value at the lower bound of the choice set
    tempMaxAPInd=ub
    tempC=s.cashInHandsGrid[j,i]-s.apRev[ub,i]
    tempV=u(m,tempC)+m.beta*s.VF.EVGrid[ub,i]
    tempMaxV=u(m,tempC)+m.beta*s.VF.EVGrid[ub,i]

    #If the lower and upper bounds for borrowing differ, iterate over other possible values of borrowing.
    if lb<ub
        for jj in (ub-1):(-1):lb

            #Calculate the value of choosing the current iteration's level of borrowing
            tempC=s.cashInHandsGrid[j,i]-s.apRev[jj,i]
            tempV=u(m,tempC)+m.beta*s.VF.EVGrid[jj,i]

            #If this value exceeds the running maximum and the current iteration's level of borrowing satisfies the arbitrarily imposed borrowing limits, update the running maximum and the policy functions
            if (tempV>tempMaxV)
                tempMaxAPInd=jj
                tempMaxV=tempV
            end
        end
    end

    tempC=s.cashInHandsGrid[j,i]-s.apRev[tempMaxAPInd,i]
    if tempC>0.0
        g2.feasibleSolution[j,i]=true
        g2.pol.apRGrid[j,i]=tempMaxAPInd
        if tempMaxV>=g2.VF.vDGrid[i]
            g2.VF.vGrid[j,i]=tempMaxV
            g2.pol.defaultGrid[j,i]=false
        else
            g2.VF.vGrid[j,i]=g2.VF.vDGrid[i]
            g2.pol.defaultGrid[j,i]=true
        end
    else
        g2.feasibleSolution[j,i]=false
        g2.VF.vGrid[j,i]=g2.VF.vDGrid[i]
        g2.pol.defaultGrid[j,i]=true
    end


    return g2
end

function updateVD!(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S},g2::onePeriodDebtUpdate{F,S}) where{F<:Real,S<:Integer}
    copyto!(g2.EVA0,s.VF.EVGrid[s.a0Ind,:])
    g2.VF.EVDGrid.=(1.0-m.theta)*s.yTMat'*s.VF.vDGrid+m.theta*g2.EVA0

    g2.VF.vDGrid.=s.VF.vDFlow.+m.beta*g2.VF.EVDGrid

    return g2
end

function updateEV!(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S},g2::onePeriodDebtUpdate{F,S}) where{F<:Real,S<:Integer}

    mul!(g2.VF.EVGrid,g2.VF.vGrid,s.yTMat)

    return g2
end

function updateQ!(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S},g2::onePeriodDebtUpdate{F,S}) where{F<:Real,S<:Integer}


    mul!(g2.qGrid,(1.0.-g2.pol.defaultGrid),s.yTMat)
    g2.qGrid.*=m.R^(-1)

    return g2
end




function updateEval!(m::onePeriodDebtSpec{F,S},s::onePeriodDebtEval{F,S},g2::onePeriodDebtUpdate{F,S}) where{F<:Real,S<:Integer}
    copyto!(s.VF,g2.VF)
    copyto!(s.pol,g2.pol)
    copyto!(s.qGrid,g2.qGrid)


    updateAPRev!(m,s)
    return s
end



#vfiGOneStep! is the workhorse function of the alternative solution method for the model which updates the bond price function after every update of the government value functions. Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model which will be modified
#3. tol: the tolerance for convergence for both the value functions and the bond price function
#4. maxIterI: the maximum number of iterations
#Its output is just the modified collection of objects used to solve the model.

function vfiGOneStep!(m::onePeriodDebtSpec{F,U},s::onePeriodDebtEval{F,U},tol::F,maxIter::U) where{F<:Real,U<:Integer}


    #Initialize new copies of the value functions, policy functions, and bond price function

    g2=makeUpdate(m,s)

    #Initialize measures of sup norm distance between various objects, a counter for the number of iterations, and a counter for the number of consecutive iterations without change in the bond price function
    iCount=1
    vDist=tol+1.0
    EVDist=tol+1.0
    vDDist=tol+1.0
    EVDDist=tol+1.0
    qDist=tol+1.0
    maxDist=tol+1.0

    #Iterate until the maximum number of iterations has been reached or the sup norm distance between successive iterations drops below the tolerance for convergence
    while (iCount<=maxIter)&(maxDist>=tol)

        updateVD!(m,s,g2)

        #Iterate over states of the income process
        Threads.@threads for i in 1:(m.yParams.yPoints)
        #for i in 1:(m.yParams.yPoints)

            #Calculate the value of repayment and corresponding policy functions for every level of incoming net assets at the current income state
            solveRepayRow!(m,s,g2,i,false)
        end

        updateEV!(m,s,g2)
        updateQ!(m,s,g2)

        #Calculate the various measures of distance between successive iterations
        vDist=maximum(abs.(g2.VF.vGrid.-s.VF.vGrid))
        EVDist=maximum(abs.(g2.VF.EVGrid.-s.VF.EVGrid))
        vDDist=maximum(abs.(g2.VF.vDGrid.-s.VF.vDGrid))
        EVDDist=maximum(abs.(g2.VF.EVDGrid.-s.VF.EVDGrid))
        qDist=maximum(abs.(g2.qGrid.-s.qGrid))
        maxDist=max(vDist,EVDist,vDDist,EVDDist,qDist)

        #At each iteration which is a multiple of 100, print the current iteration number and variables tracking the convergence criteria
        if mod(iCount,100)==0
            println([iCount,qDist,EVDist,EVDDist,vDist,vDDist])
        end

        #Increment the iteration counter
        iCount+=1

        #Update the value functions
        updateEval!(m,s,g2)
    end
    println([iCount-1,qDist,EVDist,EVDDist,vDist,vDDist])

    #Get full repayment policy functions
    #Iterate over states of the income process
    Threads.@threads for i in 1:(m.yParams.yPoints)
    #for i in 1:(m.yParams.yPoints)

        #Calculate the value of repayment and corresponding policy functions for every level of incoming net assets at the current income state
        solveRepayRow!(m,s,g2,i,true)
    end
    copyto!(s.pol,g2.pol)

    #Update the marker tracking convergence of the value functions
    if max(vDist,EVDist,vDDist,EVDDist)>tol
        s.noConvergeMark[1]=true
    else
        s.noConvergeMark[1]=false
    end

    #Return the modified collection of objects used to solve the model
    return s
end

