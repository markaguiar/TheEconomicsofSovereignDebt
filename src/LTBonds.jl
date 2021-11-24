module LTBonds 

using SpecialFunctions
using Roots
using UnPack

using Polyester 

using Random 

import Distributions: pdf
import Distributions: cdf
using Distributions 

import StatsFuns: invsqrt2π, invsqrt2

using LinearAlgebra
using LoopVectorization
using Distributions
using FastGaussQuadrature
using QuadGK

include("types.jl")  # load the types first
include("types_bonds.jl") 
include("types_models.jl") 
include("types_simulations.jl")

include("initializers.jl")
include("preferences.jl")
include("processes.jl")
include("m_shocks.jl")
include("budget_and_returns.jl")
include("bellman.jl")
include("solver.jl")
include("helper_functions.jl")
include("simulations.jl")

export 

# types 

    YProcess, 
    YDiscretized, 
    MTruncatedNormal, 
    MUniform, 
    Log,
    CRRA,
    CRRA2, 
    Preferences, 
    DefCosts, 

    AbstractBond, 
    Bond, 
    BondCE2012,
    
    LTBondModel,

    Path,
    
    make_CRRA, 
    
    get_bond, 
    get_λ, 
    get_κ,

    get_base_pars, 
    get_b_pol, 
    get_d_pol, 
    get_v, 
    get_q, 
    get_u,
    get_vD,   
    get_b_grid,
    get_y_grid,
    get_η,
    get_R,

# preferences 
    inv_u, 

# processes
    discretize,

# initializers
    generate_workspace, 

# government problem 
    solve!,
    
# helper_functions
    all_default_at,    

# budget_and_returns
    risk_free_price,
    yield, 
    find_bond_return,

# simulation 
    draws, 
    simulation, 
    simulation!,
    moments,
    create_shocks_paths

end