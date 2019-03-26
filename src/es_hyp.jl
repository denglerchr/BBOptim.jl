"""
Struct containing the constant hyperparameters for the ES algorithm:
*std::Number             # Standard deviation of the random exploration
*population_size::UInt   # Population size
*df_clip::Number         # Maximum df allowed in one perturbed direction, clamps bigger abs. values
*maxiter::UInt           # Maximal allowed number of iterations
*maxevals::Number        # Maximal allowed number of function evaluations
*npersist::UInt          # Number of persisting directions (must be <population size)
*dx_tol::Number          # Stopping criterion, stop if "gradient" of the optimisation variable (L2 norm) is less than this
*f_tol::Number           # Stopping criterion, stop if best function evaluation return a value less than this
*optimizer::AbstractOptimizer # Can be either: GradDesc, Momentum, Adam
*printiter::Int          # How often to print
"""
mutable struct EvolStrat
    std::Number             # Standard deviation of the random exploration
    population_size::UInt   # Population size
    df_clip::Number         # Maximum df allowed in one perturbed direction, clamps bigger abs. values
    maxiter::UInt           # Maximal allowed number of iterations
    maxevals::Number        # Maximal allowed number of function evaluations
    npersist::UInt          # Number of persisting directions (must be <population size)
    dx_tol::Number          # Stopping criterion, stop if "gradient" of the optimisation variable (L2 norm) is less than this
    f_tol::Number           # Stopping criterion, stop if best function evaluation return a value less than this
    optimizer::AbstractOptimizer # Can be either: GradDesc, Momentum, Adam
    printiter::UInt          # How often to print

    # Perform some checks before creating this
    function EvolStrat(std, ps, df_clip, maxiter, maxevals, npersist, dx_tol, f_tol, optimizer, printiter)
        @assert std>0
        @assert df_clip>0.0
        @assert maxiter>0
        @assert maxevals>0
        @assert npersist<=ps
        @assert dx_tol>0.0
        @assert printiter>0
        return new(std, ps, df_clip , maxiter, maxevals, npersist, dx_tol, f_tol, optimizer, printiter)
    end

end

EvolStrat(x::Vector) = EvolStrat(0.1, length(x), Inf, 1000, Inf, 0, 1e-6, -Inf, GradDesc(0.01), 1)

function show(io::IO, par::EvolStrat)
    println("Hyperparameters:")
    println("\tstd:\t\t\t $(par.std)")
    println("\tpopulation_size:\t $(par.population_size)")
    println("\tdf_clip:\t\t $(par.df_clip)")
    println("\tmaxiter:\t\t $(par.maxiter)")
    println("\tmaxevals:\t\t $(par.maxevals)")
    println("\tnpersist:\t\t $(par.npersist)")
    println("\tdx_tol:\t\t\t $(par.dx_tol)")
    println("\tf_tol:\t\t\t $(par.f_tol)")
    println("\toptimizer:\t\t $(typeof(par.optimizer))")
    println("\tprintiter:\t\t $(par.printiter)")
end
