
# generic probability calculation functions  

survival(m::MShock, x) = 1 - cdf(get_d(m), x)

mass(m::MShock, a, b) = ifelse(b > a, cdf(get_d(m), b) - cdf(get_d(m), a), zero(a))

pdf(m::MShock, x) = pdf(get_d(m), x)

cdf(m::MShock, x) = cdf(get_d(m), x)


# Specializing to MTruncatedNormal for speed 

survival(m::MTruncatedNormal, x) = 1 - cdf(m, x)

mass(m::MTruncatedNormal, a, b) = ifelse(b > a, cdf(m, b) - cdf(m, a), zero(a))

function pdf(m::MTruncatedNormal, x) 
    if x > m.m_max || x < m.m_min 
        return zero(x)
    else 
        return exp(-abs2(x / m.std)/2) * m.pdf_adjust_factor
    end 
end 

function cdf(m::MTruncatedNormal, x) 
    if x > m.m_max 
        return one(x)
    elseif x < m.m_min
        return zero(x)
    else 
        return (erfc(- x * m.cdf_1) / 2 - m.cdf_2) * m.cdf_3
    end 
end 


function discretize(y::YProcess)
    grid, Π, lr_mean = tauchen(y)
    return YDiscretized(;
        grid, Π, lr_mean,
        process = y 
    )   
end 


# Modified from: 
#   https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/markov/markov_approx.jl
function tauchen(e) 
    # Get discretized space
    σ = e.std
    ρ = e.ρ
    μ = e.μ

    y_max = e.span * sqrt(1 / (1 - ρ^2)) * σ

    if e.n == 1 
        return [1.0], [1.0], 1.0
    end
    
    y = LinRange(-y_max, y_max, e.n)
    d = (y[2] - y[1]) 

    nor = Normal(0.0, 1.0)
    std_norm_cdf = x -> cdf(nor, x)

    # Get transition probabilities
    Π = zeros(e.n, e.n)
    if e.tails == true
        for i = 1:e.n
            # Do end points first
            Π[1, i] = std_norm_cdf((y[1] - ρ*y[i] + d/2) / σ)
            Π[e.n, i] = 1 - std_norm_cdf((y[e.n] - ρ*y[i] - d/2) / σ)

            # fill in the middle columns
            for j = 2:e.n-1
                Π[j, i] = (std_norm_cdf((y[j] - ρ*y[i] + d/2) / σ) -
                               std_norm_cdf((y[j] - ρ*y[i] - d/2) / σ))
            end
        end
    else

        for i = 1:e.n
            for j = 1:(e.n)
                Π[j, i] = (std_norm_cdf((y[j] - ρ*y[i] + d/2) / σ) -
                               std_norm_cdf((y[j] - ρ*y[i] - d/2) / σ))
            end
        end

    end

    # NOTE: We need to shift this vector after finding probabilities
    #       because when finding the probabilities we use a function
    #       std_norm_cdf that assumes its input argument is distributed
    #       N(0, 1). After adding the mean E[y] is no longer 0.
    #
    y_levels = exp.(y .+ μ / (1 - ρ))
    ΠSums = sum(Π, dims=1)
    for i in 1:e.n
        Π[:, i] .*= ΠSums[i]^-1
    end
    #Since, in the long run, ln(y_t) is distributed normally with mean m=g.mu / (1 - g.rho) and variance s2=g.σ2/(1-g.rho^2),
    #y_t is distributed lognormally in the long run, and the mean of such a lognormal variable is exp(m+1/2*s2)
    y_mean = exp(μ / (1 - ρ) + 1/2 * (σ^2) / (1-ρ^2))

    return y_levels, Π, y_mean
end

