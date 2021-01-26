# This code was primarily written by Stelios Fourakis.  We are greatly indebted for his work on this project.  All errors are our own.


#This file contains a library of functions for solving the model of government default with long term debt described in Chatterjee and Eyigungor (2012).


#The code implementing Tauchen's method for approximating an AR(1) process via a Markov Chain is largely
#taken from https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/markov/markov_approx.jl

# We have also tried to use the following convention in for loops, especially in nested ones:
#1. i iterates over values of the income process
#2. j iterates over values of current period borrowing
#3. jj iterates over either values of next period borrowing or entries in the government policy functions for next period borrowing
#Except where otherwise noted, the first dimension of an array corresponds to the asset state and the second to the income process state

#Throughout, we use the convention that debt is an asset, so negative amounts indicate net liabilities.

#The code relies largely on a set of parametrically typed immutable objects which are defined at the beginning of the code. These types are:
#1. ar1Params - includes the specification of an AR(1) process as well as the range and resolution of the Markov chain to be used to approximate it
#2. iidParams - includes the specification of an iid process
#3. apSearchParams - includes the information necessary to implement a bisection style grid search which exploits the monotonicity of of the government's borrowing policy function
#4. longTermBondSpec - includes value of basic parameters of the model (beta, R, etc.), the specification of the income process, and the specification of the borrowing grid
#5. longTermBondEval - includes the income process, value function grids, policy function grids, the bond price function, and some other convenience parameters/grids


#Many of the functions below mutate the contents of one or more of their arguments.

using Distributions, QuadGK, SpecialFunctions, SparseArrays, LinearAlgebra, DelimitedFiles

#ar1Params is a parametrically typed immutable containing:
#1. yPoints: the number of points to be used in discretizing the AR(1) Process
#2. rho: the persistence parameter of the process
#3. eta2: the variance of the innovations of the process
#4. mu: the permanent component of the process
#5. stdSpan: the number of standard deviations to each side of the mean which should fall within the grid
#6. inflateEndPoints: a Boolean variable marking whether the probability of jumping to a point outside the grid should be assigned to the closest endpoint (true)
#or discarded (effectively reassigning it to the remaining points in proportion to their probability of being reached)
#Throughout, we use the convention y_t=mu+rho*y_t-1+sqrt(eta2)*e_t
struct ar1Params{F<:Real,S<:Integer}
    yPoints::S
    rho::F
    eta2::F
    mu::F
    stdSpan::F
    inflateEndPoints::Bool
end


#iidParams is a parametrically typed immutable containing:
#1. mPoints: the number of points to be used in discretizing the AR(1) Process
#2. eta2: the variance of the process
#3. mu: the mean of the process
#4. stdSpan: the number of standard deviations to each side of the mean which should fall within the grid
struct iidParams{F<:Real,S<:Integer}
    mPoints::S
    epsilon2::F
    mu::F
    stdSpan::F
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


#longTermBondSpec is a parametrically typed immutable containing enough of the information required to specify the model in Chatterjee and Eyigungor (2012) that the remainder can be fully deduced and constructed. 
#Its contents are:

#1. beta: the government's discount factor
#2. theta: the probability of reentry to international markets when in financial autarky
#3. gamma: the coefficient of relative risk aversion for CRRA preferences (which we assume both the government and consumer have, throughout)
#4. hpen0: either the h in y_def(s)=min(y(s),h*E[y(s)]), i.e. the output penalty when in default, or d0 in y_def(s)=y(s)-max(0,d0*y(s)+d1*y(s)^2)
#5. hpen1: d1 in y_def(s)=y(s)-max(0,d0*y(s)+d1*y(s)^2)
#6. R: the gross interational interest rate
#7. lambda: the maturity parameter of the debt structure
#8. coup: the coupon parameter of the debt structure
#9. aBounds: the boundaries of the borrowing grid [min, max]
#10. aPoints: the number of points to be used in the borrowing grid
#11. yParams: the specification of the persistent component of the income process and how it shall be approximated
#12. iidParams: the specification of the iid component of the income process
#13. simplePen: a boolean variable which is true if y_def(s)=min(y(s),h*E[y(s)]) and false if y_def(s)=y(s)-max(0,d0*y(s)+d1*y(s)^2)
#14. mixFacQ: the convex combination parameter in q^k+1 = mixFacQ*q^k + (1-mixFacQ)*H(q^k) where H is the operator which given government policies conditional on q^k, updates the bond price function
#15. mixFacV: the convex combination parameter in Z^k+1 = mixFacV*Z^k + (1-mixFacQ)*T(Z^k) where T is the operator which given government policies conditional on q^k, updates the expected continuation value function

#Throughout the code, the variable m is always an object of type longTermBondSpec

struct longTermBondSpec{F<:Real,S<:Integer}
    beta::F
    theta::F
    gamma::F
    hpen0::F
    hpen1::F
    R::F
    lambda::F
    coup::F
    aBounds::Array{F,1}
    aPoints::S
    yParams::ar1Params{F,S}
    mParams::iidParams{F,S}
    simplePen::Bool
    mixFacQ::F
    mixFacV::F
end

#vFunc is a parametrically typed immutable containing the flow utilities in the period of default and thereafter and
#the value function/continuation value function grids. Its contents are:
#1. vDFutFlow: a one dimensional array (axes are 1. current persistent income) containing the the expected value of the
#flow utility term conditional on the realization of the persistent component while in default
#2. vDInitFlow: a one dimensional array (axes are 1. current persistent income) containing the the value of the
#flow utility term conditional on the realization of the persistent component when the government enters the period
#with access to financial markets and then defaults
#3. vGrid: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing
#the expected values of the value function conditional on the realization of y and a, i.e. V(y,a)=E_m[W(y,m,a)]
#4. EVGrid: a two dimensional array (axes are 1. next period asset level and 2. current persistent income) containing
#the grid of the expected continuation value function Z(y,a')=E[V(y',a')|y]
#5. vDInitGrid: a one dimensional array (axes are 1. current persistent income) containing the value to the government of defaulting
#when it enters the period with access to financial markets.
#6. vDFutGrid: a one dimensional array (axes are 1. current persistent income) containing the value to the government
#when it enters the period without access to financial markets (i.e. already in default)
#7. EVDGrid: a one dimensional array (axes are 1. current persistent income) containing
#the grid of the expected continuation value function Z^D(y)=E[theta*V(y',0)+(1.0-theta)*V^D(y')|y]

struct vFunc{F<:Real}
    vDFutFlow::Array{F,1}
    vDInitFlow::Array{F,1}
    vGrid::Array{F,2}
    vDInitGrid::Array{F,1}
    vDFutGrid::Array{F,1}
    EVGrid::Array{F,2}
    EVDGrid::Array{F,1}
end
#policies is a parametrically typed immutable containing the government's policy functions. Its contents are:
#1. apLowM: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing the government's borrowing policy, conditional on repayment,
#when the m shock takes its lowest value.
#2. apHighM: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing the government's borrowing policy, conditional on repayment,
#when the m shock takes its highest value.

#3., 4., 5., and 6.  are mListLength, mAPThreshold, apPolicy, apProbability
#mListLength, mAPThreshold, and apPolicy fully describe the government's policy function conditional on repayment.

#7., 8., and 9. are defThreshold, defProb, and firstRepayInd
#When the above mentioned three three are combined additionally with defThreshold and firstRepayInd, they completely describe the government's policies,
#except in two special cases described just below

#10. and 11. are alwaysDefault and neverDefault
#these arrays contain marker variables for whether we can take shortcuts during the steps involving integration over m.

#apProbability and defProb are included for convenience. Combined with apPolicy and mListLength, they fully describe the
#exact transition probabilities implied by government policies.

#Specifically these objects are:

#3. mListLength: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing the number of distinct
#levels of asset which, conditional on repayment, the government might choose depending on the value of m. Since the rest of the government's
#policies conditional on repayment is stored in sparse matrices which are NOT zeroed at irrelevant values once the government's problem is
#solved again, we have to keep track of exactly how many cells are actually in use

#4. mAPThreshold: a one dimensional array (axes are 1. current persistent income) containing two dimensional sparse matrices.
#Each of these matrices is square NxN with N equal to the number of borrowing points (so that each column can potentially hold the longest
#possible policy function). For each of these matrices, the second axis is current asset level. Given a specific matrix and column, each cell
#from row 1 to the row indicated by the corresponding entry of mListLength indicates the upper bound of the values of m for which the government,
#conditional on repayment, chooses the borrowing policy specified in apPolicy.

#5. apPolicy: a one dimensional array (axes are 1. current persistent income) containing two dimensional sparse matrices.
#The layout of each is exactly as for those in mAPThreshold. Given the matrix for persistent income level i and column for asset level j,
#the entries of apPolicy[i][:,j] in rows 1 to the value of mListLength[j,i] indicate the policy choices associated with the thresholds in mAPThreshold.

#Combining these three allows us to say that, when persistent income level i and current asset level j, conditional on repayment,
#for m between it minimum value and the mAPThreshold[i][1,j], the government chooses apPolicy[i][1,j], and if mListLength[j,i] is strictly greater than 1,
#then it chooses apPolicy[i][jj,j] for m in between mAPThreshold[i][jj-1,j] and mAPThreshold[i][jj,j] for jj between 2 and mListLength[j,i], inclusive.

#6. apProbability: a one dimensional array (axes are 1. current persistent income) containing two dimensional sparse matrices.
#The layout of each is exactly as for those in mAPThreshold and apPolicy. It contains the probability that apPolicy[i][jj,j] is chosen
#for jj between 1 and mListLength[j,i], inclusive. These probabilities are NOT conditional on repayment. They are calculated taking into account
#the threshold level of m below which default occurs.

#7. defThreshold: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing the levels of m below
#which default occurs.

#8. defProbability: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing the probabilities of default.

#9. firstRepayInd: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing the first indices of mAPThreshold,
#apPolicy, and apProbability which is relevant for integrating the government's value function over realizations of m.

#10. alwaysDefault: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing Boolean variables indicating
#whether the government always defaults at each persistent income and asset state.

#11. neverDefault: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing Boolean variables indicating
#whether the government never defaults at each persistent income and asset state.

struct policies{F<:Real,S<:Integer}
    apLowM::Array{S,2}
    apHighM::Array{S,2}
    mListLength::Array{S,2}
    mAPThreshold::Array{SparseMatrixCSC{F,S},1}
    apPolicy::Array{SparseMatrixCSC{S,S},1}
    apProbability::Array{SparseMatrixCSC{F,S},1}
    defThreshold::Array{F,2}
    defProb::Array{F,2}
    firstRepayInd::Array{S,2}
    alwaysDefault::Array{Bool,2}
    neverDefault::Array{Bool,2}
end

#combinedIncomeProcess is a parametrically typed immutable which contains the full specification of the approximation (if any)
#of both components of the income process. Its contents are:
#1. yGrid: a one dimensional array containing the persistent component of the income process when the country is not in financial autarky
#2. yDefGrid: a one dimemsional array containing the persistent component of the income process when the country is in financial autarky
#3. yTMat: the transition matrix for the persistent component of the income process. It is produced using the convention that columns sum to 1.0
#4. yMean: the long term mean value (analytically) of the peristent component of the income process.
#5. mBounds: the minimum and maximum values assumed by the iid component of the income process.
#6. mGrid: a one dimensional array containing the grid of points which define the intervals for the m shock to be used in the integration steps
#7. mMidPoints: a one dimensional array containing the midpoints of the intervals in 6.
#8. mProb: the probability of realizing m in each interval defined in 6.
#9. mRes: the resolution of the grid in 6.
#10. mInfl: the inflation factor for m shock probabilities, i.e. 1/(CDF(m_max)-CDF(m_min))
#11. mDist: the distribution of the m shock (will be an object generated by the Distributions package)
struct combinedIncomeProcess{F<:Real,T<:Distribution{Univariate,Continuous}}
    yGrid::Array{F,1}
    yDefGrid::Array{F,1}
    yTMat::Array{F,2}
    yMean::F
    mBounds::Array{F,1}
    mGrid::Array{F,1}
    mMidPoints::Array{F,1}
    mProb::Array{F,1}
    mRes::F
    mInfl::F
    mDist::T
end

#longTermBondEval is a parametrically typed immutable which provides, in combination with a longTermBondSpec, all the objects and information required to solve
#the model of Chatterjee and Eyigungor (2008). Its contents are:
#1. income: a combined income process object described above.
#2. VF: a value function object described above.
#3. pol: a policy function object described above.
#4. aGrid: a one dimensional array containing the values of asset which the government can choose.
#5. aGridIncr: a two dimensional array (axes are 1. next period asset level and 2. current asset level)
#6. qGrid: a two dimensional array (axes are 1. next period asset level and 2. current persistent income) containing the bond price function
#7. netRevM0A0Grid: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing the level of consumption
#when the value of m is exactly zero and the government does not raise any new debt or buy back any old debt
#8. consM0Grid: a two dimensional array (axes are 1. next period asset level, 2. current asset level and 3. current persistent income level) containing the level of consumption
#when the value of m is exactly zero and the government sets a specified new level of (axes are 1. current asset level and 2. current persistent income)
#9. maxAPInd: a two dimensional array (axes are 1. current asset level and 2. current persistent income) containing the minimum index in the next period
#borrowing grid which results in strictly positive consumption under the maximum realization of m
#10. a0Ind: the index of 0.0 in the grid of (axes are 1. current asset level and 2. current persistent income) levels.
#11. noConvergeMark: a one dimensional array which simply tracks whether the model converged to the specified tolerance
#12. apSearchParams: the set of objects required for exploiting the binary monotonicity of the borrowing policy functions when finding the
#minimum and maximum values it takes for each level of income and assets

struct longTermBondEval{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    income::combinedIncomeProcess{F,T}
    VF::vFunc{F}
    pol::policies{F,S}
    aGrid::Array{F,1}
    aGridIncr::Array{F,2}
    qGrid::Array{F,2}
    netRevM0A0Grid::Array{F,2}
    consM0Grid::Array{F,3}
    maxAPInd::Array{S,2}
    a0Ind::S
    noConvergeMark::Array{Bool,1}
    apSearchParams::gridSearchParams{S}
end


#longTermBondUpdate is a parametrically typed immutable which contains the objects necessary to help solve
#the model and then calculate updated versions of the bond price function and the continuation value functions.
#Its contents are:
#1. VF: a value function object described above
#2. EVA0: a one dimensional array containing the reentry continuation value
#3. qGrid: a two dimensional array (axes are 1. next period asset level and 2. current persistent income) containing the updated bond price function
#4. qSum: a two dimensional array (axes are 1. next period asset level and 2. nexy period persistent income) containing the the value of the RHS terms
#in the functional equation describing the bond price function, i.e. qGrid[i,j]=E[1/R*qSum[i',j]|i]
#5. solveMarkH: a two dimensional array (axes are 1. current debt level and 2. current persistent income) containing Boolean variables indicating
#whether the government's problem at the highest level of the m shock has been solved during the current iteration
#6. solveMarkL: a two dimensional array (axes are 1. current debt level and 2. current persistent income) containing Boolean variables indicating
#whether the government's problem at the lowest level of the m shock has been solved during the current iteration.
#7. feasibleSolutionH: a two dimensional array (axes are 1. current debt level and 2. current persistent income) containing Boolean variables indicating
#whether the government's problem at the highest level of the m shock has been solved during the current iteration and that solution resulted in strictly positive consumption.
#8. feasibleSolutionL: a two dimensional array (axes are 1. current debt level and 2. current persistent income) containing Boolean variables indicating
#whether the government's problem at the lowest level of the m shock has been solved during the current iteration and that solution resulted in strictly positive consumption.
#9. maxAlwaysDefInd: a one dimensional array (axes are 1. current persistent income) containing the maximum asset level
#at which the government defaults under all possible realizations of the m shock

struct longTermBondUpdate{F<:Real,S<:Integer}
    VF::vFunc{F}
    EVA0::Array{F,1}
    qGrid::Array{F,2}
    qSum::Array{F,2}
    solveMarkH::Array{Bool,2}
    solveMarkL::Array{Bool,2}
    feasibleSolutionH::Array{Bool,2}
    feasibleSolutionL::Array{Bool,2}
    maxAlwaysDefInd::Array{S,1}
end


#debtDist is a parametrically typed immutable describing the stationary, ergodic joint distirbution of default state, persistent income state, and current borrowing state. Its contents are:

#1. repayDist: a two dimensional array (axes are 1. incoming borrowing, 2. persistent income state) containing the noninflated joint distribution of income and borrowing, conditional
#on the government having access to financial markets (where by noninflated, we mean that we have not divided the conditional distribution by theprobability of the condition being true)
#2. defaultDist: a one dimensional array (axes are 1. income) containing the noninflated distribution of income, conditional on the government being in default
struct debtDist{F<:Real}
    repayDist::Array{F,2}
    defaultDist::Array{F,1}
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


#Define versions of the standard normal cdf for use in the tauchen method
std_norm_cdf(x::T) where {T <: Real} = 0.5 * erfc(-x/sqrt(2))
std_norm_cdf(x::Array{T}) where {T <: Real} = 0.5 .* erfc(-x./sqrt(2))



function tauchenRev(g::ar1Params{F,S}) where{F<:Real,S<:Integer}
    # Get discretized space
    eta=sqrt(g.eta2)
    a_bar = g.stdSpan * sqrt(g.eta2 / (1 - g.rho^2))
    y = LinRange(-a_bar, a_bar, g.yPoints)
    d = y[2] - y[1]

    # Get transition probabilities
    yTMat = zeros(g.yPoints, g.yPoints)
    if g.inflateEndPoints==true
        for i = 1:g.yPoints
            # Do end points first
            yTMat[1,i] = std_norm_cdf((y[1] - g.rho*y[i] + d/2) / sqrt(g.eta2))
            yTMat[ g.yPoints,i] = 1 - std_norm_cdf((y[g.yPoints] - g.rho*y[i] - d/2) / sqrt(g.eta2))

            # fill in the middle columns
            for j = 2:g.yPoints-1
                yTMat[j,i] = (std_norm_cdf((y[j] - g.rho*y[i] + d/2) / sqrt(g.eta2)) -
                               std_norm_cdf((y[j] - g.rho*y[i] - d/2) / sqrt(g.eta2)))
            end
        end
    else

        for i = 1:g.yPoints
            for j = 1:(g.yPoints)
                yTMat[j,i] = (std_norm_cdf((y[j] - g.rho*y[i] + d/2) / sqrt(g.eta2)) -
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
        yTMat[:,i].*= yTMatSums[i]^-1
    end

    #Since, in the long run, ln(y_t) is distributed normally with mean m=g.mu / (1 - g.rho) and variance s2=g.eta2/(1-g.rho^2),
    #y_t is distributed lognormally in the long run, and the mean of such a lognormal variable is exp(m+1/2*s2)
    yMeanOut=exp(g.mu / (1 - g.rho)+1/2*g.eta2/(1-g.rho^2))

    return yOut, yTMat, yMeanOut
end


#The function makeIncomeProcess constructs a income process object based on the contents of a model specification.
#Its only argument is:
#1. m: a model specification

#Its output is simply a combinedIncomeProcess object

function makeIncomeProcess(m::longTermBondSpec{F,S}) where{F<:Real,S<:Integer}
    #Generate the grid of income values, the transition matrix, and the long run mean of the process
    yGrid,yTMat,yMean=tauchenRev(m.yParams)
    ydim=m.yParams.yPoints

    #Initialize the grid of flow income values when m=0
    yDefGrid=zeros(ydim)

    #Generate the distribution of m shock
    mDist=Distributions.Normal(m.mParams.mu,sqrt(m.mParams.epsilon2))

    #Set the bounds for the m shock
    mBounds=m.mParams.stdSpan*sqrt(m.mParams.epsilon2)*[-1.0,1.0].+m.mParams.mu

    #Create the grid of points for the m shock
    mGrid=collect(LinRange(mBounds[1],mBounds[2],m.mParams.mPoints))

    #Create the midPoints of the grids for the m shock
    mMidPoints=0.5*(mGrid[2:(m.mParams.mPoints)]+mGrid[1:(m.mParams.mPoints-1)])

    mRes=(mBounds[2]-mBounds[1])/(m.mParams.mPoints-1)

    #Create the probabilities for the intervals of the m shock
    mProb=zeros(m.mParams.mPoints-1)
    for i in 2:(m.mParams.mPoints)
        mProb[i-1]=cdf(mDist,mGrid[i])-cdf(mDist,mGrid[i-1])
    end
    tempMPSumInv=sum(mProb)^(-1)
    for i in 2:(m.mParams.mPoints)
        mProb[i-1]*=tempMPSumInv
    end


    #Calculate the inflation factor for the truncated distribution of m
    mInfl=(cdf(mDist,mBounds[2])-cdf(mDist,mBounds[1]))^(-1)

    #Fill the flow income grid while in default when m=0
    for i in 1:ydim

        #If simplePen is true, we use the penalty function of Arellano (2008). Otherwise we use the penalty function of Chatterjee and Eyigungor (2012).
        if m.simplePen==false
            yDefGrid[i]=yGrid[i]-max(0.0,m.hpen0*yGrid[i]+m.hpen1*yGrid[i]^2)
        else
            yDefGrid[i]=min(yGrid[i],m.hpen0*yMean)
        end
    end

    #Check that:
    #1. the income grid while in default is strictly positive
    #2. the income grid while in default is weakly less than the income grid while not in default
    #3. the income grid while in default is weakly monotone increasing
    #4. the income grid while in default plus the lowest level of the m shock is strictly positive

    @assert minimum(yDefGrid)>0.0
    @assert minimum(yGrid.-yDefGrid)>=0.0
    @assert minimum(yDefGrid[2:end].-yDefGrid[1:(end-1)])>=0.0
    @assert minimum(yDefGrid.+mBounds[1])>0.0

    return combinedIncomeProcess(yGrid,yDefGrid,yTMat,yMean,mBounds,mGrid,mMidPoints,mProb,mRes,mInfl,mDist)
end


#The function makeValueFunction constructs a value function object based on the contents of a model specification. Its only argument is:
#1. m: a model specification

#Its output is simply a vFunc object

function makeValueFunction(m::longTermBondSpec{F,S}) where{F<:Real,S<:Integer}
    #Set some aliases
    ydim=m.yParams.yPoints
    adim=m.aPoints
    #Initialize the value function arrays
    vGrid=zeros(adim,ydim)
    vDFutGrid=zeros(ydim)
    vDInitGrid=zeros(ydim)
    EVGrid=zeros(adim,ydim)
    EVDGrid=zeros(ydim)

    #Initialize the grids of flow utility values
    vDFutFlow=zeros(ydim)
    vDInitFlow=zeros(ydim)


    return vFunc(vDFutFlow,vDInitFlow,vGrid,vDInitGrid,vDFutGrid,EVGrid,EVDGrid)
end

#The function makePolicyGrids constructs a policy function object based on the contents of a model specification.
#Its only argument is:
#1. m: a model specification

#Its output is simply a vFunc object

function makePolicyGrids(m::longTermBondSpec{F,S}) where{F<:Real,S<:Integer}
    #Set some aliases
    ydim=m.yParams.yPoints
    adim=m.aPoints
    #Initialize the collection of objects which define the government's policy functions
    apLowM=ones(S,adim,ydim)
    apHighM=ones(S,adim,ydim)
    mListLength=zeros(S,adim,ydim)
    mAPThreshold=Array{SparseMatrixCSC{F,S},1}(undef,ydim)
    apPolicy=Array{SparseMatrixCSC{S,S},1}(undef,ydim)
    apProbability=Array{SparseMatrixCSC{F,S},1}(undef,ydim)
    defThreshold=ones(F,adim,ydim)*(m.mParams.stdSpan*sqrt(m.mParams.epsilon2)+m.mParams.mu)
    defProb=zeros(F,adim,ydim)
    firstRepayInd=ones(S,adim,ydim)
    alwaysDefault=zeros(Bool,adim,ydim)
    neverDefault=ones(Bool,adim,ydim)
    dummyF=zeros(F,adim,adim)
    dummyF[1:min(adim,10),:].=-(m.mParams.stdSpan*sqrt(m.mParams.epsilon2)+m.mParams.mu)
    sDummyF=sparse(dummyF)
    dummyS=zeros(S,adim,adim)
    dummyS[1:min(adim,10),:].=1
    sDummyS=sparse(dummyS)
    for i in 1:ydim
        mAPThreshold[i]=deepcopy(sDummyF)
        apPolicy[i]=deepcopy(sDummyS)
        apProbability[i]=deepcopy(sDummyF)
    end

    return policies(apLowM,apHighM,mListLength,mAPThreshold,apPolicy,apProbability,defThreshold,defProb,firstRepayInd,alwaysDefault,neverDefault)
end

#The function makeLTBUpdate constructs a combinedIncomeProcess object based on the contents of a model specification and eval object.
#Its arguments are:
#1. m: a model specification
#2. s: the collection of objects used to solve the model

#Its output is simply a longTermBondUpdate object

function makeLTBUpdate(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T}) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    #Set some aliases
    ydim=m.yParams.yPoints
    adim=m.aPoints

    VF=deepcopy(s.VF)
    EVA0=zeros(ydim)

    EVA0.=s.VF.EVGrid[s.a0Ind,:]
    qGrid=deepcopy(s.qGrid)
    qSum=deepcopy(s.qGrid)



    solveMarkH=zeros(Bool,adim,ydim)
    solveMarkL=zeros(Bool,adim,ydim)
    feasibleSolutionH=zeros(Bool,adim,ydim)
    feasibleSolutionL=zeros(Bool,adim,ydim)

    maxAlwaysDefInd=ones(S,ydim)

    return longTermBondUpdate(VF,EVA0,qGrid,qSum,solveMarkH,solveMarkL,feasibleSolutionH,feasibleSolutionL,maxAlwaysDefInd)
end

#The function makeEval constructs a longTermBondEval object based on the contents of a model specification. Its only argument is:
#1. m: a model specification

#Its output is simply:
#1. outputEval: a collection of objects to be used in solving the model

function makeEval(m::longTermBondSpec{F,S}) where{F<:Real,S<:Integer}

    #generate the income process, value function, and policy function objects
    income=makeIncomeProcess(m)
    VF=makeValueFunction(m)
    pol=makePolicyGrids(m)

    #Generate the grid of borrowing levels
    aGrid=collect(LinRange(m.aBounds[1],m.aBounds[2],m.aPoints))
    aGridIncr=aGrid.-(1.0-m.lambda).*aGrid'

    #Set some aliases
    ydim=m.yParams.yPoints
    adim=m.aPoints

    #Initialize the bond price function
    qGrid=zeros(adim,ydim)



    #Fill the flow utility term for the initial period of default and the expected flow utility term when the government enters the period already in default
    for i in 1:ydim
        VF.vDInitFlow[i]=u(m,income.yDefGrid[i]+income.mBounds[1])
        VF.vDFutFlow[i]=income.mInfl*quadgk(x->u(m,income.yDefGrid[i]+x)*pdf(income.mDist,x),income.mBounds[1],income.mBounds[2])[1]
    end

    #Find the location of 0.0 in the borrowing grid
    a0Ind=searchsortedfirst(aGrid,zero(F))

    #Check that the result is actually 0.0, i.e. the grid has not been misspecified as an evenly spaced grid which skips 0.0
    @assert aGrid[a0Ind]==zero(F)



    #Fill the bond price function array with the risk free value
    for i in 1:ydim
        for j in 1:adim
            qGrid[j,i]=(m.lambda+(1-m.lambda)*m.coup)/(m.R-(1.0-m.lambda))
        end
    end

    #Initialize the grids for consumption when m=0 and b'=(1-lambda)*b and for when m=0 and b' takes a specified value
    netRevM0A0Grid=zeros(adim,ydim)
    consM0Grid=zeros(adim,adim,ydim)

    #Fill the grid of consumption values when m=0 and b'=(1-lambda)*b
    for i in 1:ydim
        for j in 1:adim
            netRevM0A0Grid[j,i]=income.yGrid[i]+(m.lambda+(1.0-m.lambda)*m.coup)*aGrid[j]
        end
    end

    #Calculate conditional consumption values assuming m=0, i.e. y+(lambda+(1-lambda)*z)*a-q(y,a')*(a'-(1-lambda)*a)
    for i in 1:(m.yParams.yPoints)
        for j in 1:(m.aPoints)
            view(consM0Grid,:,j,i).=netRevM0A0Grid[j,i].-view(qGrid,:,i).*view(aGridIncr,:,j)
        end
    end


    #Initialize the array of maximum feasible choices of savings
    maxAPInd=ones(S,adim,ydim)



    #Generate the set of obects used to speed the grid search for the highest and lowest values of b' chosen at every (y,b) state
    apSearchParams=genBisectSearchGrids(adim)

    #construct and return the longTermBondEval object whose components are constructed above
    outputEval=longTermBondEval(income,VF,pol,aGrid,aGridIncr,qGrid,netRevM0A0Grid,consM0Grid,maxAPInd,a0Ind,[true],apSearchParams)



    #Fill the consumption grid conditional on m=0 only, the list of minimum indices for b', and whether the government is ever forced to default
    updateMaxAP!(m,outputEval)

    return outputEval

end



#The function updateMaxAP! is a mutating function which updates the values of maxAPInd in a longTermBondEval to reflect changes in the bond price function
#Its arguments are just
#1. m: the specification for the model
#2. s: the collection of objects used to solve the model
#Its output is simply a modified version of s
function updateMaxAP!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T}) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}

    #For each level of income and current debt, find the first level (if any exists) of next period assets at which consumption takes a strictly positive values when
    #m assumes its highest value.
    for i in 1:(m.yParams.yPoints)
        for j in 1:(m.aPoints)
            apInd=m.aPoints
            s.maxAPInd[j,i]=m.aPoints
            while apInd>1
                #if s.netRevM0A0Grid[j,i]-s.qGrid[apInd,i]*s.aGridIncr[apInd,j]+s.income.mBounds[2]>0.0
                if c(m,s,i,j,apInd,s.income.mBounds[2])>0.0
                    s.maxAPInd[j,i]=apInd
                    break
                else
                    apInd+=-1
                end
            end
        end
    end

    return s
end



#c is simply a convenience function for retrieving (the argument x denotes the value of the m shock)
#when dMark is false (so the government is not in default):
#s.consM0Grid[apInd,j,i]+x
#and when dMark is true (so the government is in default):
#s.income.yDefGrid[i]+x

function c(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S,apInd::S,x::U,dMark::Bool) where{F<:Real,U<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    if dMark==false
        return s.netRevM0A0Grid[j,i]-s.qGrid[apInd,i]*s.aGridIncr[apInd,j]+x
    else
        return s.income.yDefGrid[i]+x
    end
end

function c(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S,apInd::S,x::U) where{F<:Real,U<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}

    return s.netRevM0A0Grid[j,i]-s.qGrid[apInd,i]*s.aGridIncr[apInd,j]+x

end


function c(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S,apInd::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}

    return s.netRevM0A0Grid[j,i]-s.qGrid[apInd,i]*s.aGridIncr[apInd,j]

end


#u is the CRRA utility function. Its arguments are just:
#1. m: the model specification
#2. x: the value of consumption
function u(m::longTermBondSpec{F,S},x::U) where{F<:Real,U<:Real,S<:Integer}
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

#uInverse is the inverse of the CRRA utility function. Its arguments are just:
#1. m: the model specification
#2. x: the value of consumption

#UNLIKE THE FUNCTION JUST ABOVE, THIS FUNCTION ASSUMES THAT gamma>1.
function uInverse(m::longTermBondSpec{F,S},x::U) where{F<:Real,U<:Real,S<:Integer}
    @assert m.gamma>1 "CRRA <=1:  code assumes CRRA>1"    
    if x<0.0
        return ((1.0-m.gamma)*x)^(1.0/(1.0-m.gamma))
    else
        return uInverse(m,-1e-10)/(1.0+abs(x))
    end
end

# function uInverse(m::longTermBondSpec{F,S},x::U) where{F<:Real,U<:Real,S<:Integer}
#     if m.gamma==one(F)
#         return exp(x)
#     elseif m.gamma>one(F)
#         if x<0.0
#             return ((1.0-m.gamma)*x)^(1.0/(1.0-m.gamma))
#         else
#             return uInverse(m,-1e-10)/(1.0+abs(x))
#         end
#     else
#         if x>0.0
#             return ((1.0-m.gamma)*x)^(1.0/(1.0-m.gamma))
#         else
#             return uInverse(m,1e-10)/(1.0+abs(x))
#         end
#     end        
# end



#solveRepayChoice is a function which solve the government's problem under repayment for a specific value of persistent income, transient income, and incoming asset level.
#Its arguments are:
#1. m: the model specification
#2. s: the list of objects used to solve the model
#3. i: the index of the persistent income state
#4. j: the index of the incoming asset state
#5. x: the level of the m shock
#6. lb: the minimum index in the borrowing grid for the grid search
#7. ub: the maximum index in the borowing grid for the grid search

#This function assumes that lb<=ub so the check lb==ub is equivalent to lb<=ub. In the current implementation, it is never passed values which disobey this.
function solveRepayChoice(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S,x::F,lb::S,ub::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    #If the lower and upper bound are the same, return the that index and the corresponding value as the solution
    if lb==ub
        return lb, u(m,c(m,s,i,j,lb,x))+m.beta*s.VF.EVGrid[lb,i],c(m,s,i,j,lb,x)>zero(F)
    else
        #Generate temporary variables containing (in order):
        #1. the maximum consumption observed over levels of next period borrowing examined
        #2. the value of consumption for the level of next period borrowing currently being considered
        #3. the maximum available value thusfar observed for the government
        #4. the index of next period borrowing which results in the value of 3.

        #Initialize these to correspond to the values when the government borrows at the level indicated by lb
        maxC=c(m,s,i,j,ub,x)
        tempC=c(m,s,i,j,ub,x)
        maxVR=u(m,tempC)+m.beta*s.VF.EVGrid[ub,i]
        maxAPInd=ub

        #Iterate over the remaining values of next period borrowing
        for jj in (ub-1):-1:(lb)

            #Set the value of consumption for the level of next period borrowing currently under consideration
            tempC=c(m,s,i,j,jj,x)

            #If that value is higher than the maximum value observed thusfar, calculate the government's value under this choice
            if tempC>maxC
                tempVR=u(m,tempC)+m.beta*s.VF.EVGrid[jj,i]

                #If the value of the current choice is strictly greater than the highest value thusfar observed, set the maximum value and maximizing value variables accordingly
                if tempVR>maxVR
                    maxVR=tempVR
                    maxAPInd=jj
                end

                #Track that the maximum value of consumption observed thusfar has changed
                maxC=tempC
            end
        end

        #Return the maximizing value of the next period borrowing index, the value to the government associated with it,
        #and whether the optimal choice of asset resulted in strictly positive consumption
        return maxAPInd, maxVR,c(m,s,i,j,maxAPInd,x)>zero(F)
    end
end



#solveRepayInterior! is probably the most important function in this library. It constructs the full borrowing policy functions and
#default policy functions of the government (should the government repay under any realization of m).
#Its arguments are:
#1. m: the model specification
#2. s: the list of objects used to solve the model
#3. i: the index of the persistent income state
#4. j: the index of the incoming asset state
#5. setExactP: a Boolean variable indicating whether to calculate the exact transition probabilities implied by the policy functions

function solveRepayInterior!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S,setExactP::Bool) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    #Set aliases for the index of assets chosen when m takes its highest possible value and the index of assets chosen when m takes its highest possible value
    lb=s.pol.apLowM[j,i]
    ub=s.pol.apHighM[j,i]
    #THIS SHOULD ONLY EVERY BE INVOKED WHEN lb==ub. THE IMPLEMENTATION WHICH PRECEDES THIS IN THE MAIN FUNCTIONS SHOULD BE SUCH THAT
    #lb<=ub WHENEVER solveRepayInterior! IS CALLED.
    #IT ALSO ASSUMES THAT IT IS NEVER CALLED WHEN neverDefault[j,i] IS TRUE. FOR THE CURRENT IMPLEMENTATION WHICH PRECEDES THIS IN THE
    #MAIN FUNCTIONS, THIS IS CERTAINLY ALWAYS THE CASE.

    #If the government chooses the same value at the lowest realization of m that it does at the highest realization of m, use a special method
    #for filling in the contents of the government policy functions.
    if lb>=ub
        #In every case, we set:
        #1. the m threshold at which the government ceases to use the first element of its policy function to the maximum value of m
        #2. the borrowing policy for the first element to be the lower bound
        if s.pol.neverDefault[j,i]==true
            s.pol.mAPThreshold[i][1,j]=s.income.mBounds[2]
            s.pol.apPolicy[i][1,j]=lb
            #When the government never defaults in income/debt state (i,j), further set:
            #3. the first index of the borrowing policy function which is relevant for the integration to 1
            #4. the threshold for default to the lowest possible value of the m shock
            #5. the length of the list of thresholds for this state to 1
            s.pol.firstRepayInd[j,i]=1
            s.pol.defThreshold[j,i]=s.income.mBounds[1]

            s.pol.mListLength[j,i]=1
        else
            s.pol.mAPThreshold[i][1,j]=s.income.mBounds[2]
            s.pol.apPolicy[i][1,j]=lb
            #When the government sometimes defaults in income/debt state (i,j), further set:
            #3. the first index of the borrowing policy function which is relevant for the integration to 1
            #4. the threshold for default to uInverse(initial value of default - discounted continuation value of repayment under the optimal level of next period borrowing) minus
            #the level of consumption when m=0 under repayment at the optimal level of next period borrowing
            #5. the length of the list of thresholds for this state to 1
            s.pol.firstRepayInd[j,i]=1

            s.pol.defThreshold[j,i]=uInverse(m,s.VF.vDInitGrid[i]-m.beta*s.VF.EVGrid[lb,i])-c(m,s,i,j,lb)
            s.pol.mListLength[j,i]=1
        end
    else
        #When lb<ub, the borrowing policy function is more complicated. In this case, we initialize three temporary variables to be used in a while loop. They are:
        #1. the length of the borrowing policy function should the while loop terminate
        #2. the index of the level of borrowing which at position tempLength in the borrowing policy function for which we want to find the threshold level of m below
        #which it is chosen
        #3. the index of the level of borrowing which is chosen above the threshold level of m in 2.
        #We begin by setting the length to one, the index of level of borrowing associated with that position to index chosen at the lowest possible value
        #of the m shock, and the alternative level of borrowing to 1 less than that index
        tempLength=1
        tempOldAP=lb
        tempNewAP=lb+1
        #Note that, in the range which is used for this function ub is the unrestricted maximizer when m=m_min and lb is the unrestricted maximizer when m=m_max
        #While we are still considering levels of borrowing which fall in the relevant range, continue the loop.
        while tempNewAP<=ub

            #Get the level of consumption at the current two levels of borrowing under consideration when m=0
            tempCM0Old=c(m,s,i,j,tempOldAP)
            tempCM0New=c(m,s,i,j,tempNewAP)

            #If borrowing less results in higher consumption, immediately "remove" the "old" value from the policy function and deincrement the length variable.
            #Otherwise, proceed to find the threshold.

            #A second condition here has been added due to an edge case that arose when finding the indifference levels
            #of beta for a consumer. In general, if the bond price function has been calculated correctly, asset choice
            #should be increasing in current assets (for a fixed value of income). This assumes, however,
            #arbitrary precision of all values involved, which we do not in general
            #use when programming. Roundoff errors resulted in cases where the second condition held as true in only
            #the first iteration after specifying a new value of beta for the government. Since, if we do not include
            #the second condition (which is rather strict; note the equality requirement for consumption), tempLength-1
            #can be zero, the update to tempOldAP will throw and error. To change this, uncomment the if statement just above the specification
            #of tempEVDiff, comment out the next two lines after the following comment starting with "Calculate",
            #and uncomment the specification of tempEVDiff which occurs in the following else block.

            #if tempCM0New>=tempCM0Old
            #Calculate the term (beta*(V_Old-V_New))
            tempEVDiff=m.beta*(s.VF.EVGrid[tempOldAP,i]-s.VF.EVGrid[tempNewAP,i])
            if (tempCM0New>tempCM0Old)||((tempCM0New==tempCM0Old)&&(tempEVDiff<0.0))

                tempOldAP=s.pol.apPolicy[i][tempLength-1,j]
                tempLength+=-1

            else
                #Here we solve the equation 1/(C_New+m)-1/(C_Old+m)+beta*(V_Old-V_New)=0 for m. Some algebra yields:
                #(C_Old-C_New)/((C_New+m)*(C_Old+m))+beta*(V_Old-V_New)=0
                #(C_New+m)*(C_Old+m)+(C_Old-C_New)/(beta*(V_Old-V_New)=0
                #m^2+(CNew+C_Old)*m+C_New*C_Old+(C_Old-C_New)/(beta*(V_Old-V_New))=0
                #This is a parabola which opens up. We are interested in the rightmost root (the other one is, I believe, in general below whatever previous threshold was found)
                #We use the quadratic formula to do so below.

                #Calculate the term (beta*(V_Old-V_New))
                #tempEVDiff=m.beta*(s.VF.EVGrid[tempOldAP,i]-s.VF.EVGrid[tempNewAP,i])

                #tempEVDiff should, in general always be negative (if Z(y,a') is increasing in a'). If it is 0.0, set it to a negative value of very small magnitude.
                if tempEVDiff==0.0
                    tempEVDiff=-eps(F)
                end

                #Calculate the threshold value of m at which the government is indifferent between to the two levels of borrowing
                tempMThres=-0.5*(tempCM0New+tempCM0Old)+sqrt((tempCM0New-tempCM0Old)/tempEVDiff-tempCM0New*tempCM0Old+0.25*(tempCM0New+tempCM0Old)^2)

                #If tempLength is 1 (and therefore tempOldAP=lb; this will be the case even when we are not in the first iteration of the while loop):
                #1. Set the first entry of the threshold component of the borrowing policy function appropriately
                #2. Set the first entry of the borrowing policy function indices appropriately
                #3. Set the index for which we want to determine the upper bound of m values such that it is chosen to the index of the value for which we just found the threshold
                #4. Set the index of the borrowing value which should be used as the alternative in the indifference calculation for the next threshold to be one less than the index in 3.
                #5. Increment the length counter
                if tempLength==1
                    s.pol.mAPThreshold[i][tempLength,j]=tempMThres
                    s.pol.apPolicy[i][tempLength,j]=tempOldAP
                    tempOldAP=tempNewAP
                    tempNewAP=tempOldAP+1
                    tempLength+=1
                else
                    #If the borrowing policy function currently contains more entries than just what occurs at the lowest realization of m, we first check whether the threshold just
                    #calculated is less than the threshold above which tempOldAP SHOULD be chosen. If that is the case, then tempOldAP is NEVER chosen. We set tempOldAP to the value of
                    #the entry just above it in the borrowing policy function and deincrement the length counter (so that tempOldAP's presence in the borrowing policy function will
                    #eventually be overwritten and replaced with the proper value). Note that in this case tempNewAP is not changed.
                    if tempMThres<s.pol.mAPThreshold[i][tempLength-1,j]
                        tempOldAP=s.pol.apPolicy[i][tempLength-1,j]
                        tempLength+=-1
                    else
                        #If the new threshold is weakly greater than the old one:
                        #1. we add the upper bound of m values at which tempOldAP is chosen to the borrowing policy function
                        #2. record that it is chosen in the relevant interval
                        #3. make the new index for which an upper bound is to be determined the one which was used to find the upper bound in 1.
                        #4. set the index which we will attempt to used to find an upper bound for the m values at which the index in #3. is chosen to 1 less than the index in #3.
                        #5. increment the length counter.
                        s.pol.mAPThreshold[i][tempLength,j]=tempMThres
                        s.pol.apPolicy[i][tempLength,j]=tempOldAP
                        tempOldAP=tempNewAP
                        tempNewAP=tempOldAP+1
                        tempLength+=1
                    end
                end
            end
        end
        #When the above while loop exits, it will always be the case that tempOldAP=lb (since lb was the unrestricted maximizer at m=m_max) and tempNewAP
        #was strictly less than lb. We know the upper bound of values at which lb is chosen, and we know that it IS chosen. Record those facts, and record that
        #the first tempLength indices of the borrowing policy function are relevant for the current iteration's solution
        s.pol.mAPThreshold[i][tempLength,j]=s.income.mBounds[2]
        s.pol.apPolicy[i][tempLength,j]=ub
        s.pol.mListLength[j,i]=tempLength


        #We now move on determining default thresholds, if necessary
        if s.pol.neverDefault[j,i]==true
            #If the government never defaults:
            #1. set the default threshold to the lower bound of the m shock
            #2. set the value of the first index of the borrowing policy function relevant for the integration steps to 1
            #3. set the probability of default to 0
            #4. calculate the probabilities associated with each choice of next period borrowing
            s.pol.defThreshold[j,i]=s.income.mBounds[1]
            s.pol.firstRepayInd[j,i]=1

        else
            #When the government does default for some values of m but not for others, determine which of the intervals specified by s.income.mBounds and s.pol.mAPThreshold it defaults in
            #Initialize this index at one
            defIdx=1
            #Continue until we reach the last relevant index
            #At each step, check whether the value at the upper bound of m values associated with that interval is weakly greater than the value of default. Once this is true, break the loop.
            #If it is false, increment the index of the interval in which we need to check whether default occurs
            #THESE STEPS ALL ASSUME THAT THIS SECTION IS NEVER CALLED WHEN EITHER alwaysDefault[j,i] OR neverDefault[j,i] ARE TRUE.
            while defIdx<=s.pol.mListLength[j,i]
                if u(m,c(m,s,i,j,s.pol.apPolicy[i][defIdx,j],s.pol.mAPThreshold[i][defIdx,j]))+m.beta*s.VF.EVGrid[s.pol.apPolicy[i][defIdx,j],i]>=s.VF.vDInitGrid[i]
                    break
                else
                    defIdx+=1
                end
            end

            #Set the index of the first element of the borrowing policy function relevant for the integration steps to the index of the interval in which default occurs
            s.pol.firstRepayInd[j,i]=defIdx

            #Calculate the threshold at which default occurs within that interval
            s.pol.defThreshold[j,i]=uInverse(m,s.VF.vDInitGrid[i]-m.beta*s.VF.EVGrid[s.pol.apPolicy[i][defIdx,j],i])-c(m,s,i,j,s.pol.apPolicy[i][defIdx,j])

        end
    end

    #If exact transition probabilities have been requested, calculate them.
    if setExactP==true
        calculateProbabilities!(m,s,i,j)
    end

    return s
end



#calculateProbabilities! determines the exact probabilities of state transitions implied by government policy functions.
#Note that it assumes that alwaysDefault for the state in question is false.
#Its arguments are:
#1. m: the model specification
#2. s: the list of objects used to solve the model
#3. i: the index of the persistent income state
#4. j: the index of the incoming asset state

function calculateProbabilities!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}

    #If the policy function only has one entry, use a simpler method of calculating transition probabilities
    #Otherwise perform the full procedure of assigning probabilities (some possibly zero) to all outcomes
    #possible according to the current policy function
    if s.pol.mListLength[j,i]==1
        if s.pol.neverDefault[j,i]==true

            #Set the probability of default to 0
            s.pol.defProb[j,i]=0.0

            #Set the probability of choosing the first component of the borrowing policy function to 100%
            s.pol.apProbability[i][1,j]=1.0

        else
            #Set the probability of default to the value implied by that threshold
            s.pol.defProb[j,i]=s.income.mInfl*(cdf(s.income.mDist,s.pol.defThreshold[j,i])-cdf(s.income.mDist,s.income.mBounds[1]))
            #Set the probability of choosing the first component to the complement of the value just calculated
            s.pol.apProbability[i][1,j]=1.0-s.pol.defProb[j,i]
        end
    else
        #Initialize a variable tracking the sum of all noninflated probabilities calculated so that we can normalize exactly before exiting.
        #It is not uncommon for the sum to be slightly different from one if we simply multiply everything by s.income.mInfl
        tempPSum=0.0
        if s.pol.neverDefault[j,i]==true
            s.pol.defProb[j,i]=0.0

            #Set the probability associated with the first element of the borrowing policy function using the lower bound for the m shock
            #and the upper bound given by the thresholds calculated above
            s.pol.apProbability[i][1,j]=(cdf(s.income.mDist,s.pol.mAPThreshold[i][1,j])-cdf(s.income.mDist,s.income.mBounds[1]))

            #Set the remaining probabilities using the difference in the CDF between adjacent thresholds
            for jj in 2:(s.pol.mListLength[j,i])
                s.pol.apProbability[i][jj,j]=(cdf(s.income.mDist,s.pol.mAPThreshold[i][jj,j])-cdf(s.income.mDist,s.pol.mAPThreshold[i][jj-1,j]))
            end
            #Calculate the sum of the probabilities calculated, get its reciprocal, and multiply all probabilities by it
            @inbounds @simd for jj in 1:(s.pol.mListLength[j,i])
                tempPSum+=s.pol.apProbability[i][jj,j]
            end
            tempPSumInv=tempPSum^(-1)
            for jj in 1:(s.pol.mListLength[j,i])
                s.pol.apProbability[i][jj,j]*=tempPSumInv
            end
        else
            #Set an alias for the index of the policy function thresholds where the default decision changes
            defIdx=s.pol.firstRepayInd[j,i]
            #Set the probability of default appropriately
            s.pol.defProb[j,i]=(cdf(s.income.mDist,s.pol.defThreshold[j,i])-cdf(s.income.mDist,s.income.mBounds[1]))

            #Add the probability of default to the variable tracking the sum of all probabilities calculated
            tempPSum+=s.pol.defProb[j,i]
            #Set the probability associated with the full borrowing policy function appropriately. To do this, we set the probability associated with any index
            #strictly less than that of the interval in which default occurs to 0.0, set the probability associated with that index to only the change in the CDF
            #of m between the default threshold and the upper bound of m values at which that choice is made, and then, if necessary, set any remaining probabilities
            #by using the change in the CDF of m between adjacent thresholds.
            if defIdx==1
                s.pol.apProbability[i][1,j]=(cdf(s.income.mDist,s.pol.mAPThreshold[i][1,j])-cdf(s.income.mDist,s.pol.defThreshold[j,i]))
                for jj in 2:(s.pol.mListLength[j,i])
                    s.pol.apProbability[i][jj,j]=(cdf(s.income.mDist,s.pol.mAPThreshold[i][jj,j])-cdf(s.income.mDist,s.pol.mAPThreshold[i][jj-1,j]))
                end
                #Calculate the sum of the probabilities calculated, get its reciprocal, and multiply all probabilities by it
                @inbounds @simd for jj in 1:(s.pol.mListLength[j,i])
                    tempPSum+=s.pol.apProbability[i][jj,j]
                end
                tempPSumInv=tempPSum^(-1)
                s.pol.defProb[j,i]*=tempPSumInv

                for jj in 1:(s.pol.mListLength[j,i])
                    s.pol.apProbability[i][jj,j]*=tempPSumInv
                end
            else
                s.pol.apProbability[i][1:(defIdx-1),j].=0.0
                s.pol.apProbability[i][defIdx,j]=(cdf(s.income.mDist,s.pol.mAPThreshold[i][defIdx,j])-cdf(s.income.mDist,s.pol.defThreshold[j,i]))

                if defIdx<s.pol.mListLength[j,i]
                    for jj in (defIdx+1):(s.pol.mListLength[j,i])
                        s.pol.apProbability[i][jj,j]=(cdf(s.income.mDist,s.pol.mAPThreshold[i][jj,j])-cdf(s.income.mDist,s.pol.mAPThreshold[i][jj-1,j]))
                    end
                end
                #Calculate the sum of the probabilities calculated, get its reciprocal, and multiply all probabilities by it
                @inbounds @simd for jj in defIdx:(s.pol.mListLength[j,i])
                    tempPSum+=s.pol.apProbability[i][jj,j]
                end
                tempPSumInv=tempPSum^(-1)

                s.pol.defProb[j,i]*=tempPSumInv
                for jj in defIdx:(s.pol.mListLength[j,i])
                    s.pol.apProbability[i][jj,j]*=tempPSumInv
                end
            end
        end
    end
    return s
end

#solveRepayRow! is one of the workhorse functions of this library. It solves the government's problem under repayment for every possible incoming level of borrowing
#given a specific value for the persistent component of income. It takes as its arguments:
#1. m: a model specification
#2. s: the collection of objects used to solve the model
#3. g2: the collection of objects used, along with the policy functions in s, to solve the model. update the guesses of q and Z
#4. i: the index of the value of the persistent component of income
#5. setExactP: a Boolean variable indicating whether or not to calculate exact transition probabilities

#Its output is just modified versions of s and g2

function solveRepayRow!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},g2::longTermBondUpdate{F,S},i::S,setExactP::Bool) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}

    tempAInd=1

    #initialize temporary variables for the upper and lower bounds of the grid search as well as the index of current asset holdings
    tempLB=1
    tempUB=s.maxAPInd[1,i]

    #Solve the government's problem at the lowest value of incoming borrowing under the lowest m shock
    s.pol.apLowM[1,i],tempVLow,g2.feasibleSolutionL[1,i]=solveRepayChoice(m,s,i,1,s.income.mBounds[1],tempLB,tempUB)

    #Mark that the problem has indeed been solved at the lowest value of incoming borrowing under the lowest m shock
    g2.solveMarkL[1,i]=true

    if g2.feasibleSolutionL[1,i]==true
        tempLB=s.pol.apLowM[1,i]
    end

    s.pol.apHighM[1,i],tempVHigh,g2.feasibleSolutionH[1,i]=solveRepayChoice(m,s,i,1,s.income.mBounds[2],tempLB,tempUB)

    #Mark that the problem has indeed been solved at the lowest value of incoming borrowing under the highest m shock
    g2.solveMarkH[1,i]=true

    #Set the alwaysDefault and neverDefault variables appropriately
    if tempVLow>=s.VF.vDInitGrid[i]
        s.pol.neverDefault[1,i]=true
        s.pol.alwaysDefault[1,i]=false
    elseif tempVHigh<s.VF.vDInitGrid[i]
        s.pol.neverDefault[1,i]=false
        s.pol.alwaysDefault[1,i]=true
    else
        s.pol.neverDefault[1,i]=false
        s.pol.alwaysDefault[1,i]=false
    end


    #reinitialize temporary variables for the upper and lower bounds of the grid search
    tempLB=1
    tempUB=s.maxAPInd[m.aPoints,i]

    #Perform the exact same steps for the highest level of incoming borrowing
    s.pol.apLowM[m.aPoints,i],tempVLow,g2.feasibleSolutionL[m.aPoints,i]=solveRepayChoice(m,s,i,m.aPoints,s.income.mBounds[1],tempLB,tempUB)

    g2.solveMarkL[m.aPoints,i]=true


    if g2.feasibleSolutionL[m.aPoints,i]==true
        tempLB=s.pol.apLowM[m.aPoints,i]
    end

    s.pol.apHighM[m.aPoints,i],tempVHigh,g2.feasibleSolutionH[m.aPoints,i]=solveRepayChoice(m,s,i,m.aPoints,s.income.mBounds[2],tempLB,tempUB)

    g2.solveMarkH[m.aPoints,i]=true

    if tempVLow>=s.VF.vDInitGrid[i]
        s.pol.neverDefault[m.aPoints,i]=true
        s.pol.alwaysDefault[m.aPoints,i]=false
    elseif tempVHigh<s.VF.vDInitGrid[i]
        s.pol.neverDefault[m.aPoints,i]=false
        s.pol.alwaysDefault[m.aPoints,i]=true
    else
        s.pol.neverDefault[m.aPoints,i]=false
        s.pol.alwaysDefault[m.aPoints,i]=false
    end



    #Loop over the number of levels in the search parameters
    for levelInd in 1:(s.apSearchParams.numLevels)
        #Loop over the number of points in the current level
        for pointInd in 1:(s.apSearchParams.levelPoints[levelInd])
            #Set the index of the point currently under consideration
            tempAInd=s.apSearchParams.pointGrids[levelInd][pointInd]
            #If the problem has not been solved at this index for the highest realization of m, solve it. Otherwise (i.e. if the government was found to have defaulted at a
            #lower debt level under the highest realization of m), skip solving it, since the result does not matter.
            if g2.solveMarkH[tempAInd,i]==false

                #Get the indices associated with points for which the problem has already been solved whose solutions, if feasible, form upper and lower bounds
                #for the solution associated with the current index.
                tempUBInd=s.apSearchParams.pointUB[levelInd][pointInd]
                tempLBInd=s.apSearchParams.pointLB[levelInd][pointInd]

                #Check whether a feasible solution was found at those two points. If so, use the best bound available. Otherwise, use the alternative bounds already known.
                tempUB=ifelse(g2.feasibleSolutionH[tempUBInd,i]==true,min(s.maxAPInd[tempAInd,i],s.pol.apHighM[tempUBInd,i]),s.maxAPInd[tempAInd,i])
                tempLB=ifelse(g2.feasibleSolutionH[tempLBInd,i]==true,s.pol.apHighM[tempLBInd,i],1)

                #Solve the problem, check whether the solution resulted in strictly positive consumption, and mark that we have solved the problem at the current index
                s.pol.apHighM[tempAInd,i],tempVHigh,g2.feasibleSolutionH[tempAInd,i]=solveRepayChoice(m,s,i,tempAInd,s.income.mBounds[2],tempLB,tempUB)

                g2.solveMarkH[tempAInd,i]=true

                #If the value to the government under the highest realization of the m shock is strictly less than the initial value of default, then this will be true for:
                #1. all lower values of the m shock at this level of income and incoming asset.
                #2. all higher values of incoming asset at this level of income and the highest m shock.
                #3. all higher values of incoming asset at any level of the m shock (applying 1 to the states in 2 yields this result)
                #We use this information to mark that we know government default policy at such points (and because the government always defaults do not need
                #to solve the government's problem under repayment for any value of the m shock.
                #We also update the minAlwaysDefInd variable so that, should this condition be true for another value of debt, we need not update values which have already been set correctly.

                #If the value to the government under the highest realization of the m shock is weakly greater than the initial value of default, then we know that there are at least some values of
                #m at which the government does not default, so we set the always default variable appropriately.
                if tempVHigh<s.VF.vDInitGrid[i]
                    s.pol.neverDefault[g2.maxAlwaysDefInd[i]:tempAInd,i].=false
                    s.pol.alwaysDefault[g2.maxAlwaysDefInd[i]:tempAInd,i].=true
                    g2.solveMarkH[g2.maxAlwaysDefInd[i]:tempAInd,i].=true
                    g2.solveMarkL[g2.maxAlwaysDefInd[i]:tempAInd,i].=true
                    g2.maxAlwaysDefInd[i]=max(g2.maxAlwaysDefInd[i],tempAInd)
                else
                    s.pol.alwaysDefault[tempAInd,i]=false
                end
            end
            #If the problem has not been solved at this index for the lowest realization of m, solve it. Otherwise (i.e. if the government was found to have defaulted at a
            #lower debt level under the highest realization of m), skip solving it, since the result does not matter.
            if g2.solveMarkL[tempAInd,i]==false
                #Get the indices associated with points for which the problem has already been solved whose solutions, if feasible, form upper and lower bounds
                #for the solution associated with the current index.
                tempUBInd=s.apSearchParams.pointUB[levelInd][pointInd]
                tempLBInd=s.apSearchParams.pointLB[levelInd][pointInd]

                #Check whether a feasible solution was found at those two points. If so, use the best bound available. Otherwise, use the alternative bounds already known.
                tempUB=ifelse(g2.feasibleSolutionL[tempUBInd,i]==true,min(s.pol.apLowM[tempUBInd,i],s.maxAPInd[tempAInd,i]),s.maxAPInd[tempAInd,i])
                tempLB=ifelse(g2.feasibleSolutionL[tempLBInd,i]==true,s.pol.apLowM[tempLBInd,i],1)

                #Solve the problem, check whether the solution resulted in strictly positive consumption, and mark that we have solved the problem at the current index
                s.pol.apLowM[tempAInd,i],tempVLow,g2.feasibleSolutionL[tempAInd,i]=solveRepayChoice(m,s,i,tempAInd,s.income.mBounds[1],tempLB,tempUB)

                g2.solveMarkL[tempAInd,i]=true

                #Check whether the value to the government under the lowest possible value of the m shock is strictly greater than the initial value of default.
                #If this is the case, then the government never defaults in this state, and we can set the neverDefault variable to true. Otherwise, there are at least some values
                #of m at which the government defaults, so we set it to false.
                if tempVLow>=s.VF.vDInitGrid[i]
                    s.pol.neverDefault[tempAInd,i]=true
                else
                    s.pol.neverDefault[tempAInd,i]=false
                end
            end
        end
    end

    #Loop over levels of debt.
    for j in 1:m.aPoints
        #Whenever the government does not always default (so we care about its full policy functions), construct the full policy function and probability objects associated with that state.
        if s.pol.alwaysDefault[j,i]==false
            solveRepayInterior!(m,s,i,j,setExactP)
        end
    end


    return s,g2

end




#integrateMVExact performs the integration step in V(y,b)=E_m[W(y,m,b)] for a specific value of the persistent component of income and incoming asset, given government policies.
#This implementation uses numerical quadrature to perform the integration. Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. i: the index of the persistent component of income
#4. j: the index of the incoming level of borrowing
#It returns a single number which is the value of the integral.
function integrateMVExact(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    #Initialize the output variable
    outValV=0.0

    #If the government always defaults in this state, return just the initial value of default. If the government never defaults, we know that we can disregard the
    #possible contribution of the initial value of default to the value function and must integrate over the m intervals associated with all possible borrowing policies.
    #If neither of these is the case, we fall back to the foolproof procedure of adding the contribution of the initial value, then adding the value associated with any intervals
    #in which the government does not default.
    if s.pol.alwaysDefault[j,i]==true
        outValV+=s.VF.vDInitGrid[i]
    elseif s.pol.neverDefault[j,i]==true
        #Add the value of flow utility in the first m interval to the output variable
        outValV+=s.income.mInfl*quadgk(x->u(m,c(m,s,i,j,s.pol.apPolicy[i][1,j],x))*pdf(s.income.mDist,x),s.income.mBounds[1],s.pol.mAPThreshold[i][1,j])[1]
        #If there are more intervals, add the flow values associated with them
        if s.pol.mListLength[j,i]>1
            for jj in 2:(s.pol.mListLength[j,i])
                outValV+=s.income.mInfl*quadgk(x->u(m,c(m,s,i,j,s.pol.apPolicy[i][jj,j],x))*pdf(s.income.mDist,x),s.pol.mAPThreshold[i][jj-1,j],s.pol.mAPThreshold[i][jj,j])[1]
            end
        end
        #Add the discounted expected value contribution associated with every m interval
        @inbounds @simd for jj in 1:(s.pol.mListLength[j,i])
        #for jj in 1:(s.pol.mListLength[j,i])
            outValV+=m.beta*s.pol.apProbability[i][jj,j]*s.VF.EVGrid[s.pol.apPolicy[i][jj,j],i]
        end
    else
        #Add to our output variable with the contribution of initial default value
        outValV+=s.pol.defProb[j,i]*s.VF.vDInitGrid[i]

        #Get the index of the first cell in the borrowing policy function which occurs with strictly positive probability
        startInd=s.pol.firstRepayInd[j,i]

        #Add the flow utility contribution in the interval of m realizations which range from the default threshold level to the upper bound at which this first cell is relevant
        outValV+=s.income.mInfl*quadgk(x->u(m,c(m,s,i,j,s.pol.apPolicy[i][startInd,j],x))*pdf(s.income.mDist,x),s.pol.defThreshold[j,i],s.pol.mAPThreshold[i][startInd,j])[1]

        #If there are more cells of the borrowing policy function which are relevant, add their flow utility contributions
        if (s.pol.mListLength[j,i])>(startInd)
            for jj in (startInd+1):(s.pol.mListLength[j,i])
                outValV+=s.income.mInfl*quadgk(x->u(m,c(m,s,i,j,s.pol.apPolicy[i][jj,j],x))*pdf(s.income.mDist,x),s.pol.mAPThreshold[i][jj-1,j],s.pol.mAPThreshold[i][jj,j])[1]
            end
        end

        #Add the discounted expected value contributions for every relevant interval of the borrowing policy function.
        @inbounds @simd for jj in (startInd):(s.pol.mListLength[j,i])
        #for jj in (startInd):(s.pol.mListLength[j,i])
            outValV+=m.beta*s.pol.apProbability[i][jj,j]*s.VF.EVGrid[s.pol.apPolicy[i][jj,j],i]
        end
    end
    return outValV
end



#integrateMVApprox performs the integration step in V(y,b)=E_m[W(y,m,b)] for a specific value of the persistent component of income and incoming asset, given government policies.
#This function performs the integration exactly as described in Chatterjee and Eyigungor (2012). Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. i: the index of the persistent component of income
#4. j: the index of the incoming level of borrowing
#It returns a single number which is the value of the integral.

#The structure and internal logic of this function are essentially identical to those of the one directly above it.
function integrateMVApprox(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    outValV=0.0
    if s.pol.alwaysDefault[j,i]==true
        outValV+=s.VF.vDInitGrid[i]
    elseif s.pol.neverDefault[j,i]==true

        #If the government never defaults, then we begin the integration over the result under repayment at the first interval.
        #We set the location of the current upper bound in the m grid to the second point, the index of the relevant borrowing policy entry to the first,
        #and the most recent value of m to its lower bound
        mUBInd=2
        aPolicyInd=1
        lastMVal=s.income.mGrid[1]
        #Loop until we exit the upper boundary of the m space (the second check is to ensure that we never exit the range of the current borrowing policy function;
        #it should never matter, and if it does, the process should in general not lead to convergence).
        while (mUBInd<=(m.mParams.mPoints))&&(aPolicyInd<=(s.pol.mListLength[j,i]))
            #If the upper bound for the range in which the current entry of the borrowing policy function is valid is strictly less than the upper bound of the current
            #m interval, add the relevant contribution of that entry, update the last value of m reached, and increment only the variable marking our position in the borrowing
            #policy function
            if s.pol.mAPThreshold[i][aPolicyInd,j]<(s.income.mGrid[mUBInd])
                outValV+=s.income.mProb[mUBInd-1]*(s.pol.mAPThreshold[i][aPolicyInd,j]-lastMVal)/s.income.mRes*(u(m,c(m,s,i,j,s.pol.apPolicy[i][aPolicyInd,j],s.income.mMidPoints[mUBInd-1]))+m.beta*s.VF.EVGrid[s.pol.apPolicy[i][aPolicyInd,j],i])
                lastMVal=s.pol.mAPThreshold[i][aPolicyInd,j]
                aPolicyInd+=1
            else
                #Otherwise, add the relevant contribution of the current entry which lies in this m interval, update the last value of m reached to be its upper bound,
                #and then increment only the variable marking our position in the m grid
                outValV+=s.income.mProb[mUBInd-1]*(s.income.mGrid[mUBInd]-lastMVal)/s.income.mRes*(u(m,c(m,s,i,j,s.pol.apPolicy[i][aPolicyInd,j],s.income.mMidPoints[mUBInd-1]))+m.beta*s.VF.EVGrid[s.pol.apPolicy[i][aPolicyInd,j],i])
                lastMVal=s.income.mGrid[mUBInd]

                mUBInd+=1
            end
        end

    else
        #If the government does default only sometimes, find the position in the m grid of the upper bound of the interval in which the threshold level of m for default falls
        #and set the last value of m observed to the value directly preceding that upper bound
        mUBInd=searchsortedfirst(s.income.mGrid,s.pol.defThreshold[j,i])
        lastMVal=s.income.mGrid[mUBInd-1]
        #If there are any entire intervals in which default occurs, add their contribution to the output value
        if mUBInd>2
            for k in 3:mUBInd
                outValV+=s.income.mProb[k-2]*s.VF.vDInitGrid[i]

            end
        end
        #Set the index of the first relevant entry of the borowing policy function
        aPolicyInd=s.pol.firstRepayInd[j,i]

        #Add the contribution of default in the interval in which the threshold lies to the output value
        outValV+=s.income.mProb[mUBInd-1]*(s.pol.defThreshold[j,i]-lastMVal)/s.income.mRes*s.VF.vDInitGrid[i]

        #Update the last value of m observed to the threshold level of m at which default occurs
        lastMVal=s.pol.defThreshold[j,i]
        #Loop until we exit the upper boundary of the m space (the second check is to ensure that we never exit the range of the current borrowing policy function;
        #it should never matter, and if it does, the process should in general not lead to convergence).
        while (mUBInd<=(m.mParams.mPoints))&&(aPolicyInd<=(s.pol.mListLength[j,i]))
            #If the upper bound for the range in which the current entry of the borrowing policy function is valid is strictly less than the upper bound of the current
            #m interval, add the relevant contribution of that entry, update the last value of m reached, and increment only the variable marking our position in the borrowing
            #policy function
            if s.pol.mAPThreshold[i][aPolicyInd,j]<(s.income.mGrid[mUBInd])
                outValV+=s.income.mProb[mUBInd-1]*(s.pol.mAPThreshold[i][aPolicyInd,j]-lastMVal)/s.income.mRes*(u(m,c(m,s,i,j,s.pol.apPolicy[i][aPolicyInd,j],s.income.mMidPoints[mUBInd-1]))+m.beta*s.VF.EVGrid[s.pol.apPolicy[i][aPolicyInd,j],i])
                lastMVal=s.pol.mAPThreshold[i][aPolicyInd,j]
                aPolicyInd+=1
            else
                #Otherwise, add the relevant contribution of the current entry which lies in this m interval, update the last value of m reached to be its upper bound,
                #and then increment only the variable marking our position in the m grid
                outValV+=s.income.mProb[mUBInd-1]*(s.income.mGrid[mUBInd]-lastMVal)/s.income.mRes*(u(m,c(m,s,i,j,s.pol.apPolicy[i][aPolicyInd,j],s.income.mMidPoints[mUBInd-1]))+m.beta*s.VF.EVGrid[s.pol.apPolicy[i][aPolicyInd,j],i])
                lastMVal=s.income.mGrid[mUBInd]
                mUBInd+=1
            end
        end
    end

    return outValV
end



#integrateMVAlmostExact performs the integration step in V(y,b)=E_m[W(y,m,b)] for a specific value of the persistent component of income and incoming asset, given government policies.
#This function performs the integration almost exactly as described in Chatterjee and Eyigungor (2012).
#It differs from their method in that it applies exactly their assumption that m is uniformly distributed in each subinterval and takes advantage of that to perform the integration analytically.
#Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. i: the index of the persistent component of income
#4. j: the index of the incoming level of borrowing
#It returns a single number which is the value of the integral.

#The structure and internal logic of this function are essentially identical to those of the one directly above it.
function integrateMVAlmostExact(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    outValV=0.0
    if s.pol.alwaysDefault[j,i]==true
        outValV+=s.VF.vDInitGrid[i]
    elseif s.pol.neverDefault[j,i]==true

        #If the government never defaults, then we begin the integration over the result under repayment at the first interval.
        #We set the location of the current upper bound in the m grid to the second point, the index of the relevant borrowing policy entry to the first,
        #and the most recent value of m to its lower bound
        mUBInd=2
        aPolicyInd=1
        lastMVal=s.income.mGrid[1]
        tempCBase=0.0
        tempCHigh=0.0
        tempCLow=0.0
        #Loop until we exit the upper boundary of the m space (the second check is to ensure that we never exit the range of the current borrowing policy function;
        #it should never matter, and if it does, the process should in general not lead to convergence).
        while (mUBInd<=(m.mParams.mPoints))&&(aPolicyInd<=(s.pol.mListLength[j,i]))
            #If the upper bound for the range in which the current entry of the borrowing policy function is valid is strictly less than the upper bound of the current
            #m interval, add the relevant contribution of that entry, update the last value of m reached, and increment only the variable marking our position in the borrowing
            #policy function
            if s.pol.mAPThreshold[i][aPolicyInd,j]<(s.income.mGrid[mUBInd])
                #tempCBase=(s.netRevM0A0Grid[j,i]-s.qGrid[s.pol.apPolicy[i][aPolicyInd,j],i]*s.aGridIncr[s.pol.apPolicy[i][aPolicyInd,j],j])
                tempCBase=c(m,s,i,j,s.pol.apPolicy[i][aPolicyInd,j])
                tempCLow=tempCBase+lastMVal
                tempCHigh=tempCBase+s.pol.mAPThreshold[i][aPolicyInd,j]
                outValV+=s.income.mProb[mUBInd-1]*((log(tempCLow)-log(tempCHigh))/s.income.mRes+(s.pol.mAPThreshold[i][aPolicyInd,j]-lastMVal)/s.income.mRes*m.beta*s.VF.EVGrid[s.pol.apPolicy[i][aPolicyInd,j],i])

                lastMVal=s.pol.mAPThreshold[i][aPolicyInd,j]
                aPolicyInd+=1
            else
                #Otherwise, add the relevant contribution of the current entry which lies in this m interval, update the last value of m reached to be its upper bound,
                #and then increment only the variable marking our position in the m grid
                #tempCBase=(s.netRevM0A0Grid[j,i]-s.qGrid[s.pol.apPolicy[i][aPolicyInd,j],i]*s.aGridIncr[s.pol.apPolicy[i][aPolicyInd,j],j])
                tempCBase=c(m,s,i,j,s.pol.apPolicy[i][aPolicyInd,j])
                tempCLow=tempCBase+lastMVal
                tempCHigh=tempCBase+s.income.mGrid[mUBInd]
                outValV+=s.income.mProb[mUBInd-1]*((log(tempCLow)-log(tempCHigh))/s.income.mRes+(s.income.mGrid[mUBInd]-lastMVal)/s.income.mRes*m.beta*s.VF.EVGrid[s.pol.apPolicy[i][aPolicyInd,j],i])

                lastMVal=s.income.mGrid[mUBInd]

                mUBInd+=1
            end
        end

    else
        #If the government does default only sometimes, find the position in the m grid of the upper bound of the interval in which the threshold level of m for default falls
        #and set the last value of m observed to the value directly preceding that upper bound
        mUBInd=searchsortedfirst(s.income.mGrid,s.pol.defThreshold[j,i])
        lastMVal=s.income.mGrid[mUBInd-1]
        #If there are any entire intervals in which default occurs, add their contribution to the output value
        if mUBInd>2
            for k in 3:mUBInd
                outValV+=s.income.mProb[k-2]*s.VF.vDInitGrid[i]

            end
        end
        #Set the index of the first relevant entry of the borowing policy function
        aPolicyInd=s.pol.firstRepayInd[j,i]

        #Add the contribution of default in the interval in which the threshold lies to the output value
        outValV+=s.income.mProb[mUBInd-1]*(s.pol.defThreshold[j,i]-lastMVal)/s.income.mRes*s.VF.vDInitGrid[i]

        #Update the last value of m observed to the threshold level of m at which default occurs
        lastMVal=s.pol.defThreshold[j,i]
        tempCBase=0.0
        tempCHigh=0.0
        tempCLow=0.0
        #Loop until we exit the upper boundary of the m space (the second check is to ensure that we never exit the range of the current borrowing policy function;
        #it should never matter, and if it does, the process should in general not lead to convergence).
        while (mUBInd<=(m.mParams.mPoints))&&(aPolicyInd<=(s.pol.mListLength[j,i]))
            #If the upper bound for the range in which the current entry of the borrowing policy function is valid is strictly less than the upper bound of the current
            #m interval, add the relevant contribution of that entry, update the last value of m reached, and increment only the variable marking our position in the borrowing
            #policy function
            if s.pol.mAPThreshold[i][aPolicyInd,j]<(s.income.mGrid[mUBInd])
                #tempCBase=(s.netRevM0A0Grid[j,i]-s.qGrid[s.pol.apPolicy[i][aPolicyInd,j],i]*s.aGridIncr[s.pol.apPolicy[i][aPolicyInd,j],j])
                tempCBase=c(m,s,i,j,s.pol.apPolicy[i][aPolicyInd,j])
                tempCLow=tempCBase+lastMVal
                tempCHigh=tempCBase+s.pol.mAPThreshold[i][aPolicyInd,j]
                outValV+=s.income.mProb[mUBInd-1]*((log(tempCLow)-log(tempCHigh))/s.income.mRes+(s.pol.mAPThreshold[i][aPolicyInd,j]-lastMVal)/s.income.mRes*m.beta*s.VF.EVGrid[s.pol.apPolicy[i][aPolicyInd,j],i])

                lastMVal=s.pol.mAPThreshold[i][aPolicyInd,j]
                aPolicyInd+=1
            else
                #Otherwise, add the relevant contribution of the current entry which lies in this m interval, update the last value of m reached to be its upper bound,
                #and then increment only the variable marking our position in the m grid
                #tempCBase=(s.netRevM0A0Grid[j,i]-s.qGrid[s.pol.apPolicy[i][aPolicyInd,j],i]*s.aGridIncr[s.pol.apPolicy[i][aPolicyInd,j],j])
                tempCBase=c(m,s,i,j,s.pol.apPolicy[i][aPolicyInd,j])
                tempCLow=tempCBase+lastMVal
                tempCHigh=tempCBase+s.income.mGrid[mUBInd]
                outValV+=s.income.mProb[mUBInd-1]*((log(tempCLow)-log(tempCHigh))/s.income.mRes+(s.income.mGrid[mUBInd]-lastMVal)/s.income.mRes*m.beta*s.VF.EVGrid[s.pol.apPolicy[i][aPolicyInd,j],i])

                lastMVal=s.income.mGrid[mUBInd]
                mUBInd+=1
            end
        end
    end

    return outValV
end



#integrateMQSumExact performs the first step of the integration in the functional equation which q must satisfy for a single value of the persistent component of income
#and incoming borrowing level. It returns the term E_m[(1-d(y,m,b))*(lambda+(1-lambda)(coup+q(y,m,b'(y,m,b))))]. This implementation uses the exact probabilities
#implied by the threshold values of m to calculate that value. Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. i: the index of the persistent component of income
#4. j: the index of the incoming level of borrowing
#It returns a single number which is the value of the integral.
function integrateMQSumExact(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    #Initialize the output value
    outValQ=0.0

    #If the government ever repays in the given state, sum the appropriate values
    if s.pol.alwaysDefault[j,i]==false
        #Add the portion of the price which depends only on the default decision
        outValQ+=(1.0-s.pol.defProb[j,i])*(m.lambda+(1.0-m.lambda)*m.coup)
        #Add the remaining portions of the price
        @inbounds @simd for jj in (s.pol.firstRepayInd[j,i]):(s.pol.mListLength[j,i])
        #for jj in (s.pol.firstRepayInd[ii,j]):(s.pol.mListLength[ii,j])
            #outValQ+=s.pol.apProbability[i][jj,j]*(m.lambda+(1.0-m.lambda)*(m.coup+s.qGrid[s.pol.apPolicy[i][jj,j],i]))
            outValQ+=s.pol.apProbability[i][jj,j]*(1.0-m.lambda)*s.qGrid[s.pol.apPolicy[i][jj,j],i]
        end
    end

    return outValQ

end




#integrateMQSumApprox performs the first step of the integration in the functional equation which q must satisfy for a single value of the persistent component of income
#and incoming borrowing level. It returns the term E_m[(1-d(y,m,b))*(lambda+(1-lambda)(coup+q(y,m,b'(y,m,b))))]. This implementation uses exactly the approximation
#described in Chatterjee and Eyigungor (2012). Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. i: the index of the persistent component of income
#4. j: the index of the incoming level of borrowing
#It returns a single number which is the value of the integral.

#The structure and internal logic of this function is essentially identical to that of integrateMVApprox
function integrateMQSumApprox(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},i::S,j::S) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    #Initialize the output value
    outValQ=0.0

    #If the government ever repays in the given state, sum the appropriate values
    if s.pol.neverDefault[j,i]==true
        mUBInd=2
        aPolicyInd=1
        lastMVal=s.income.mGrid[1]
        while (mUBInd<=(m.mParams.mPoints))&&(aPolicyInd<=(s.pol.mListLength[j,i]))
            if (s.pol.mAPThreshold[i][aPolicyInd,j])<(s.income.mGrid[mUBInd])
                outValQ+=s.income.mProb[mUBInd-1]*(s.pol.mAPThreshold[i][aPolicyInd,j]-lastMVal)/s.income.mRes*(m.lambda+(1.0-m.lambda)*(m.coup+s.qGrid[s.pol.apPolicy[i][aPolicyInd,j],i]))
                lastMVal=s.pol.mAPThreshold[i][aPolicyInd,j]
                aPolicyInd+=1
            else
                outValQ+=s.income.mProb[mUBInd-1]*(s.income.mGrid[mUBInd]-lastMVal)/s.income.mRes*(m.lambda+(1.0-m.lambda)*(m.coup+s.qGrid[s.pol.apPolicy[i][aPolicyInd,j],i]))
                lastMVal=s.income.mGrid[mUBInd]
                mUBInd+=1
            end
        end
    elseif s.pol.alwaysDefault[j,i]==false

        aPolicyInd=s.pol.firstRepayInd[j,i]
        lastMVal=s.pol.defThreshold[j,i]

        mUBInd=searchsortedfirst(s.income.mGrid,lastMVal)

        while (mUBInd<=(m.mParams.mPoints))&&(aPolicyInd<=(s.pol.mListLength[j,i]))
            if (s.pol.mAPThreshold[i][aPolicyInd,j])<(s.income.mGrid[mUBInd])
                outValQ+=s.income.mProb[mUBInd-1]*(s.pol.mAPThreshold[i][aPolicyInd,j]-lastMVal)/s.income.mRes*(m.lambda+(1.0-m.lambda)*(m.coup+s.qGrid[s.pol.apPolicy[i][aPolicyInd,j],i]))
                lastMVal=s.pol.mAPThreshold[i][aPolicyInd,j]
                aPolicyInd+=1
            else
                outValQ+=s.income.mProb[mUBInd-1]*(s.income.mGrid[mUBInd]-lastMVal)/s.income.mRes*(m.lambda+(1.0-m.lambda)*(m.coup+s.qGrid[s.pol.apPolicy[i][aPolicyInd,j],i]))
                lastMVal=s.income.mGrid[mUBInd]
                mUBInd+=1
            end
        end
    end

    return outValQ

end


#updateVD! is simple function which just updates the value function associated with default.
#Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. g2: the collection of objects used, along with the policy functions in s, to solve the model. update the guesses of q and Z

#It returns a modified version of g2
function updateVD!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},g2::longTermBondUpdate{F,S}) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}

    g2.VF.vDFutGrid.=s.VF.vDFutFlow.+m.beta*m.theta*g2.EVA0.+m.beta*(1.0-m.theta)*s.income.yTMat'*s.VF.vDFutGrid
    g2.VF.EVDGrid.=m.theta*g2.EVA0.+(1.0-m.theta)*s.income.yTMat'*g2.VF.vDFutGrid
    g2.VF.vDInitGrid.=s.VF.vDInitFlow.+m.beta*g2.VF.EVDGrid

    return g2
end


#updateV! performs the full integration across states to calculate V(y,b)=E_m[W(y,m,b)]. Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. g2: the collection of objects used, along with the policy functions in s, to solve the model. update the guesses of q and Z
#4. vIntExact: a true/false variable indicating whether quadrature or midpoint integration is to be used. True corresponds to quadrature

#Its output is simply a modified version of vNew
function updateV!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},g2::longTermBondUpdate{F,S},vIntExact::Bool)  where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    if vIntExact==true
        Threads.@threads for i in 1:(m.yParams.yPoints)
        #for i in 1:(m.yParams.yPoints)
            for j in 1:(m.aPoints)
                g2.VF.vGrid[j,i]=integrateMVExact(m,s,i,j)
            end
        end
    else
        Threads.@threads for i in 1:(m.yParams.yPoints)
            for j in 1:(m.aPoints)
                g2.VF.vGrid[j,i]=integrateMVApprox(m,s,i,j)
            end
        end

    end
    return g2
end

#updateEV! performs the second integration step in updating the main value function. Specifically, it calculates Z(y,b')=E_y'[V(y',b')].
#Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. g2: the collection of objects used, along with the policy functions in s, to solve the model. update the guesses of q and Z
function updateEV!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},g2::longTermBondUpdate{F,S})  where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}
    mul!(g2.VF.EVGrid,g2.VF.vGrid,s.income.yTMat)
    return g2
end


#updateQ! performs the full integration across states to calculate q(y,b'). Its arguments are:
#1. m: a model specification
#2. s: a collection of objects used to solve the model
#3. g2: the collection of objects used, along with the policy functions in s, to solve the model. update the guesses of q and Z
#4. vIntExact: a true/false variable indicating whether quadrature or midpoint integration is to be used. True corresponds to quadrature

#Its output is simply a modified version of qNew
function updateQ!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},g2::longTermBondUpdate{F,S},qIntExact::Bool) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}

    #Perform the first step of the integration
    if qIntExact==true
        Threads.@threads for i in 1:(m.yParams.yPoints)
        #for i in 1:(m.yParams.yPoints)
            for j in 1:(m.aPoints)
                g2.qSum[j,i]=integrateMQSumExact(m,s,i,j)/m.R
            end
        end
    else
        Threads.@threads for i in 1:(m.yParams.yPoints)
        #for i in 1:(m.yParams.yPoints)
            for j in 1:(m.aPoints)
                g2.qSum[j,i]=integrateMQSumApprox(m,s,i,j)/m.R
            end
        end
    end

    #Perform the second step of the integration, and store the result in qNew before returning it
    mul!(g2.qGrid,g2.qSum,s.income.yTMat)

    #copyto!(g2.qGrid,(g2.qSum*s.income.yTMat))
    return g2
end



#vfiGOneStep! is the main function of the for solving method for the model in Chatterjee and Eyigungor (2012)
#1. m: a model specification
#2. s: a collection of objects used to solve the model which will be modified
#3. tol: the tolerance for convergence for both the value functions and the bond price function
#4. maxIter: the maximum number of iterations
#5. vIntExact: a true/false variable indicating whether quadrature or midpoint integration is to be used. True corresponds to quadrature

#Its output is simply a modified version of s
function vfiGOneStep!(m::longTermBondSpec{F,S},s::longTermBondEval{F,S,T},tol::F,maxIter::S,vIntExact::Bool) where{F<:Real,S<:Integer,T<:Distribution{Univariate,Continuous}}




    #Check which method of integration has been requested and update the expected value of flow utility under default appropriately
    if vIntExact==false
        for i in 1:(m.yParams.yPoints)
            s.VF.vDFutFlow[i]=0.0
            for k in 1:(m.mParams.mPoints-1)
                #s.VF.vDFutFlow[i]+=s.income.mProb[k]*(log(s.income.yDefGrid[i]+s.income.mGrid[k])-log(s.income.yDefGrid[i]+s.income.mGrid[k+1]))/s.income.mRes
                s.VF.vDFutFlow[i]+=s.income.mProb[k]*u(m,s.income.yDefGrid[i]+s.income.mMidPoints[k])
            end
        end
    else
        for i in 1:(m.yParams.yPoints)
            s.VF.vDFutFlow[i]=s.income.mInfl*quadgk(x->u(m,s.income.yDefGrid[i]+x)*pdf(s.income.mDist,x),s.income.mBounds[1],s.income.mBounds[2])[1]
        end
    end

    #Initialize new copies of the value functions and the bond price function
    g2=makeLTBUpdate(m,s)


    #Initialize  a counter for the number of iterations and measures of sup norm distance between various objects
    iCount=1
    vFDist=tol+1.0
    vDFutDist=tol+1.0
    vDInitDist=tol+1.0
    EVDist=tol+1.0
    EVDDist=tol+1.0
    vDist=tol+1.0
    qDist=tol+1.0
    maxDist=tol+1.0

    #Iterate until the maximum number of iterations has been reached or the sup norm distance between successive iterations drops below the tolerance for convergence
    while (iCount<=maxIter)&(maxDist>=tol)

        #Update the default value functions and calculate the distance between the update and the previous version
        updateVD!(m,s,g2)
        vDFutDist=maximum(abs.(g2.VF.vDFutGrid-s.VF.vDFutGrid))
        EVDDist=maximum(abs.(g2.VF.EVDGrid-s.VF.EVDGrid))
        vDInitDist=maximum(abs.(g2.VF.vDInitGrid-s.VF.vDInitGrid))

        #Store the updated guesses for the default value functions
        s.VF.vDFutGrid.=g2.VF.vDFutGrid
        s.VF.EVDGrid.=g2.VF.EVDGrid
        s.VF.vDInitGrid.=g2.VF.vDInitGrid

        #Reset all the objects used to speed up the grid search step
        g2.solveMarkL.=false
        g2.solveMarkH.=false
        g2.feasibleSolutionL.=false
        g2.feasibleSolutionH.=false
        g2.maxAlwaysDefInd.=one(S)


        #Iterate over states of the persistent component of income
        Threads.@threads for i in 1:(m.yParams.yPoints)
        #for i in 1:(m.yParams.yPoints)
            #solve the problem at the current state of the persistent component of income
            solveRepayRow!(m,s,g2,i,vIntExact)
        end

        #Perform the integration steps to obtain new versions of V, Z, and q
        updateV!(m,s,g2,vIntExact)
        updateEV!(m,s,g2)
        updateQ!(m,s,g2,vIntExact)


        #Calculate the measures of distance between successive iterations for V, Z, and q
        vFDist=maximum(abs.(g2.VF.vGrid-s.VF.vGrid))
        EVDist=maximum(abs.(g2.VF.EVGrid-s.VF.EVGrid))
        qDist=maximum(abs.(g2.qGrid-s.qGrid))

        #Update the guesses for V, Z, and q
        s.VF.vGrid.=g2.VF.vGrid
        s.VF.EVGrid.=m.mixFacV*s.VF.EVGrid.+(1.0-m.mixFacV)*g2.VF.EVGrid
        s.qGrid.=m.mixFacQ*s.qGrid.+(1.0-m.mixFacQ)*g2.qGrid

        #Update the value of reentry
        g2.EVA0.=s.VF.EVGrid[s.a0Ind,:]
        #g2.EVA0.=view(s.VF.EVGrid,s.a0Ind,:)



        #Calculate the maximum of the various distance measures for the value functions
        vDist=max(vFDist,EVDist,EVDDist,vDFutDist,vDInitDist)

        #Calculate the maximum of all the various distance measures
        maxDist=max(vDist,qDist)

        #At each iteration which is a multiple of 5% of the maximum number of iterations, print the current iteration number and variables tracking the convergence criteria
        if div(iCount,max(div(maxIter,20),1))==(iCount/max(div(maxIter,20),1))
            println([iCount,qDist,EVDist,vFDist,EVDDist,vDFutDist,vDInitDist])
        end

        #Increment the iteration counter
        iCount+=1

        #Update the grid of consumption values conditional on income, incoming borrowing, and next period borrowing
        updateMaxAP!(m,s)


    end
    #Calculate conditional consumption values assuming m=0, i.e. y+(lambda+(1-lambda)*z)*a-q(y,a')*(a'-(1-lambda)*a)
    for i in 1:(m.yParams.yPoints)
        for j in 1:(m.aPoints)
            view(s.consM0Grid,:,j,i).=s.netRevM0A0Grid[j,i].-view(s.qGrid,:,i).*view(s.aGridIncr,:,j)
        end
    end


    #Calculate the exact transition probabilities associated with the final set of policy functions
    Threads.@threads for i in 1:(m.yParams.yPoints)
    #for i in 1:(m.yParams.yPoints)
        for j in 1:(m.aPoints)
            if s.pol.alwaysDefault[j,i]==false
                calculateProbabilities!(m,s,i,j)
            end
        end
    end


    #Print final set of convergence statistics
    println([iCount-1,qDist,EVDist,vFDist,EVDDist,vDFutDist,vDInitDist])

    #Update the marker tracking convergence of the value functions
    if vDist>tol
        s.noConvergeMark[1]=true
    else
        s.noConvergeMark[1]=false
    end


    #Return the modified collection of objects used to solve the model
    return s
end


