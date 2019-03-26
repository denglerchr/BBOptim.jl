# Try to find "argmin_x f(x)" using ES
# Added some heuristics:
# Exploration/Exploitation: directions with large df will persist in the seed_id
# Use f(x) also, instead of only the perturbed ones?
function es_opt(f::Function, x0::Vector, hyp_par::EsHyperparam; prefunc::Function = x->nothing, vec_workers::Vector = workers())

    # Preallocate variables
    x = copy(x0) # Current best parameters
    best_x = copy(x0) # Overall best parameters
    costlog = zeros(hyp_par.maxiter)

    #Creating remote channels for eventual parallel execution
    fitness_channel = RemoteChannel(()->Channel{Tuple{Int, Float64, Float64, Int}}(length(vec_workers)+4)) #holds the fitnesses and the corresponding seeds
    seed_channel = RemoteChannel(()->Channel{UInt32}(length(vec_workers)+4)) #holds the seed_id used by the workers to generate gaussian noise

    # Main loop
    for iter = 1:hyp_par.maxiter

        # Run preiteration-function, can be used e.g. for extra logging, running validation set, virtual batch normalization etc.
        prefunc(x)

        # Get gradient
        grad = calc_grad(f, x, hyp_par, fitness_channel, seed_channel, vec_workers)

        # Apply Optimizer
        optimize!(x, grad, hyp_par.optimizer)
        costlog[iter] = f(x)

        # Check for stopping criteria and print information
        if iter % hyp_par.printiter == 0
            println("Total costs after $iter iterations: $(costlog[iter])")
        end
        if costlog[iter]<hyp_par.f_tol
            println("Stopping because function value below tolerance")
            break
        end
        if norm(grad, 2)<hyp_par.dx_tol
            println("Stopping because gradient norm below tolerance")
            break
        end
    end

    return x, costlog
end

# Convenience wrapper #TODO test this
function es_opt(f::Function,
    x0::Vector;
    std::Number = 1.0,
    df_clip = Inf,
    population_size::UInt = 2*length(x0),
    maxiter::Int = 1000,
    maxevals = Inf,
    dx_tol::Float64 = 1e-8,
    f_tol = -Inf,
    optimizer::AbstractOptimizer = GradDesc(0.01),
    printiter::Int = 1,
    vec_workers::Vector = workers())

    # Fill hyperparameters in a struct and call main function
    hyp_par = EsHyperparam(lr, std, population_size, maxiter, maxevals, npersist, dx_tol, f_tol, optimizer,printiter)
    return es_opt(f, x0, hyp_par; vec_workers = vec_workers)
end


function calc_grad(f::Function, x::Vector{T}, hyp_par::EsHyperparam, fitness_channel, seed_channel, vec_workers) where {T}
    # Preallocate Output
    grad = zeros(T, length(x))

    nremotecalls = ceil(Int, hyp_par.population_size/2)

    # Create problem on every worker
    for (i, p) in enumerate(vec_workers)
        remote_do(EvolStrat.worker_es, p, fitness_channel, seed_channel, x, f, hyp_par.std)
    end

    @sync begin
        #generate and push all the seeds to the seed channel
        for i=1:nremotecalls
            seed_id = floor(UInt32, (typemax(UInt32)-UInt32(1))*rand())
            @async put!(seed_channel, seed_id) #In fact, only the arguments of the srand() function are pushed to the channel. For a given argument, the srand() function always returns the same set of random numbers
        end

        # One episode/ one update of theta
        for j = 1:nremotecalls
            p, F1, F2, seed_id = take!(fitness_channel)
            F = clamp(F1-F2, -hyp_par.df_clip, hyp_par.df_clip) # TODO handle NaN or Inf
            noise = hyp_par.std*randn(srand(seed_id), length(grad))
            grad .+= (F/hyp_par.population_size).*noise
        end

        # # Tell the workers that they are done
        for j = 1:length(vec_workers)
            @async put!(seed_channel, typemax(UInt32))
        end

    end

    return grad

end


function worker_es(fitness_channel::RemoteChannel, seed_channel::RemoteChannel, x::Vector, f::Function, std::Number)
    #start work
    while true
        seed_id = take!(seed_channel)

        if (seed_id == typemax(UInt32))
            return nothing
        else
            #generate random seed and calculate noise
            noise = std*randn(srand(seed_id), length(x))

            #evaluate fitness
            F1 = f(x+noise)
            F2 = f(x-noise)

            #push results to channel
            put!(fitness_channel,(myid(), F1, F2, seed_id))
        end
    end
    return nothing
end
