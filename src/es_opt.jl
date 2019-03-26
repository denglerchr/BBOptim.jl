# Try to find "argmin_x f(x)" using ES
# Added some heuristics:
# Exploration/Exploitation: directions with large df will persist in the seed_id
# Use f(x) also, instead of only the perturbed ones?
function optimize(f::Function, x0::Vector, hyp_par::EvolStrat; prefunc::Function = x->nothing, workerpool::AbstractWorkerPool = default_worker_pool())

    # Preallocate variables
    x = copy(x0) # Current best parameters
    best_x = copy(x0) # Overall best parameters
    best_f = Inf # Overall best function value
    best_seeds = [get_seed() for i = 1:hyp_par.npersist] # Contains the best seeds of the previous run
    costlog = zeros(hyp_par.maxiter)
    nevals = 0

    #Creating remote channels for eventual parallel execution TODO, can this parallel stuff be iplemented smarter? pmap?
    vec_workers = workerpool.workers
    isempty(vec_workers) ? vec_workers = workers() : nothing
    fitness_channel = RemoteChannel(()->Channel{Tuple{Int, Float64, Float64, Int}}(length(vec_workers)+4)) #holds the fitnesses and the corresponding seeds
    seed_channel = RemoteChannel(()->Channel{UInt32}(length(vec_workers)+4)) #holds the seed_id used by the workers to generate gaussian noise

    # Main loop
    for iter = 1:hyp_par.maxiter

        # Run preiteration-function, can be used e.g. for extra logging, running validation set, virtual batch normalization etc.
        prefunc(x)

        # Get gradient
        grad = calc_grad(f, x, hyp_par, best_seeds, fitness_channel, seed_channel, vec_workers)
        nevals += ceil(Int, hyp_par.population_size/2)*2

        # Apply Optimizer
        optimize!(x, grad, hyp_par.optimizer, f)
        costlog[iter] = f(x)

        # Check if we have new best parameters
        if costlog[iter]<best_f
            best_f = costlog[iter]
            best_x .= x
        end

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
        if nevals > hyp_par.maxevals
            println("Stopping because maximum allowed number of function evaluations reached")
            break
        end
    end

    return best_x, costlog
end


function calc_grad(f::Function, x::Vector{T}, hyp_par::EvolStrat, best_seeds::Vector, fitness_channel, seed_channel, vec_workers) where {T}
    # Preallocate Output
    grad = zeros(T, length(x))

    nremotecalls = ceil(Int, hyp_par.population_size/2)
    seeds = Array{UInt32}(undef, nremotecalls)
    dfs = Array{Float64}(undef, nremotecalls)

    # Create problem on every worker
    for (i, p) in enumerate(vec_workers)
        remote_do(worker_es, p, fitness_channel, seed_channel, x, f, hyp_par.std)
    end

    @sync begin
        #generate and push all the seeds to the seed channel
        for i=1:nremotecalls
            if i<=length(best_seeds)
                seed_id = best_seeds[i]
            else
                seed_id = get_seed()
            end
            @async put!(seed_channel, seed_id) #In fact, only the arguments of the srand() function are pushed to the channel. For a given argument, the srand() function always returns the same set of random numbers
        end

        # One episode/ one update of theta
        for j = 1:nremotecalls
            p, F1, F2, seed_id = take!(fitness_channel)
            F = clamp(F1-F2, -hyp_par.df_clip, hyp_par.df_clip) # TODO handle NaN or Inf
            noise = hyp_par.std*randn(Random.seed!(seed_id), length(grad))
            grad .+= (F/hyp_par.population_size).*noise
            seeds[j] = seed_id # Save seed (for sorting best seeds later)
            dfs[j] = abs(F1-F2)/norm(noise) # Save grad
        end

        # Tell the workers that they are done
        for j = 1:length(vec_workers)
            @async put!(seed_channel, typemax(UInt32))
        end

        # Sort seeds and overwrite best_seeds
        decreasing_ind = sortperm(dfs; rev = true)
        #println(dfs[decreasing_ind])
        for i = 1:length(best_seeds)
            best_seeds[i] = seeds[decreasing_ind[i]]
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
            noise = std*randn(Random.seed!(seed_id), length(x))

            #evaluate fitness
            F1 = f(x+noise)
            F2 = f(x-noise)

            #push results to channel
            put!(fitness_channel,(myid(), F1, F2, seed_id))
        end
    end
    return nothing
end


"""
Returns a random UInt32 between 0 and 0xfffffffe
"""
@inline function get_seed()
    return floor(UInt32, (typemax(UInt32)-UInt32(1))*rand())
end
