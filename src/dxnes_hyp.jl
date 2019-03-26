mutable struct Dxnes
    std::Number             # Standard deviation of the random exploration
    population_size::UInt   # Population size
    maxiter::UInt           # Maximal allowed number of iterations
    maxevals::Number        # Maximal allowed number of function evaluations
    #dx_tol::Number          # Stopping criterion, stop if "gradient" of the optimisation variable (L2 norm) is less than this
    f_tol::Number           # Stopping criterion, stop if best function evaluation return a value less than this
    #optimizer::AbstractOptimizer # Can be either: GradDesc, Momentum, Adam
    printiter::UInt          # How often to print
    std_tol::Number         # Stopping criterion, stop if std too small

    function Dxnes(std::Number, population_size::Number, maxiter::Number, maxevals::Number, f_tol::Number, printiter::Number, std_tol::Number)
        @assert std > 0.0
        @assert population_size > 0
        @assert maxiter > 0
        @assert maxevals > 0
        @assert printiter > 0
        @assert std_tol > 0
        return new(std, population_size, maxiter, maxevals, f_tol, printiter, std_tol)
    end
end


Dxnes(x::Vector) = Dxnes(1.0, 2*length(x), 1000, Inf, -Inf, 1, 1e-6)


function show(io::IO, par::Dxnes)
    println("Hyperparameters:")
    println("\tstd:\t\t\t $(par.std)")
    println("\tpopulation_size:\t $(par.population_size)")
    println("\tmaxiter:\t\t $(par.maxiter)")
    println("\tmaxevals:\t\t $(par.maxevals)")
    println("\tf_tol:\t\t\t $(par.f_tol)")
    println("\tprintiter:\t\t $(par.printiter)")
    println("\tstd_tol:\t\t $(par.std_tol)")
end
