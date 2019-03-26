mutable struct CMAES
    std::Number             # Standard deviation of the random exploration at the beginning
    population_size::UInt   # Population size
    maxiter::UInt           # Maximal allowed number of iterations
    maxevals::Number        # Maximal allowed number of function evaluations
    std_tol::Number         # Stopping criterion, stop if the standard deviation is less than this
    f_tol::Number           # Stopping criterion, stop if best function evaluation return a value less than this
    printiter::UInt          # How often to print

    function CMAES(std, population_size, maxiter, maxevals, std_tol, f_tol, printiter)
        @assert std > 0
        @assert population_size > 0
        @assert maxiter > 0
        @assert maxevals > 0
        @assert std_tol > 0
        @assert printiter > 0

        return new(std, population_size, maxiter, maxevals, std_tol, f_tol, printiter)
    end
end


CMAES(x::Vector) = CMAES(1.0, Int(4 + floor(Int, 3*log(length(x)))), 1000, Inf, 1e-6, -Inf, 1)

function show(io::IO, par::CMAES)
    println("Hyperparameters:")
    println("\tstd:\t\t\t $(par.std)")
    println("\tpopulation_size:\t $(par.population_size)")
    println("\tmaxiter:\t\t $(par.maxiter)")
    println("\tmaxevals:\t\t $(par.maxevals)")
    println("\tstd_tol:\t\t $(par.std_tol)")
    println("\tf_tol:\t\t\t $(par.f_tol)")
    println("\tprintiter:\t\t $(par.printiter)")
end
