function minimize(f::Function, x0::Vector{T}, hyp_par::Dxnes; prefunc::Function = x->nothing, workerpool::AbstractWorkerPool = default_worker_pool()) where {T}
    # Preallocate variables
    x_mean = copy(x0) # Mean of the MV Gaussian distribution
    best_x = copy(x0) # Overall best parameters
    length_x = length(x0)
    bestfitness_ever = Inf # Overall best function value
    costlog = zeros(hyp_par.maxiter)
    nevals = 0
    half_population_size = floor(Int, hyp_par.population_size/2)
    B = I
    Z = Array{T}(undef, length_x, 2*half_population_size)
    Zsorted = Array{T}(undef, length_x, 2*half_population_size)
    X = Array{T}(undef, length_x, 2*half_population_size)
    xviews = Vector{AbstractVector{T}}(undef, 2*half_population_size)

    # Learning rates
    etaMu = one(T)
    etaSigmaset = T[1.0, 0.5*(1.0 + half_population_size/(half_population_size + length_x)),  1.0 + half_population_size/(half_population_size + length_x)]
    etaBset = T[(half_population_size + length_x)/(half_population_size + length_x^2 + 50) .* min(1, sqrt(2*half_population_size/length_x)),  half_population_size/ (half_population_size+ length_x^2 + 50)]

    # Utility function
    weights = max.(zero(T), log.(T(half_population_size + 1)) .- log.(1:T(half_population_size*2)))
    uRank = weights./(sum(weights)) .- one(T)/(half_population_size*2)
    alpha::T = (0.9+0.15*log(length_x)) #*min(1.0, 2*half_population_size/length_x) # This is only in the paper, not in the matlab implementation

    # Evolution path
    base::T = sqrt(length_x) * (1 - 1/(4*length_x) + 1/(21*length_x^2));
    mueff::T = sum(weights)^2/sum(abs2, weights)
    csigma::T = (mueff + 2.0) / (length_x + mueff + 5.0) / sqrt(length_x);
    psigma = zeros(T, length_x)
    GM = zeros(T, length_x, length_x)
    GB = Symmetric(GM) # used at the very end

    # Main loop
    for iter = 1:hyp_par.maxiter

        # Run preiteration-function, can be used e.g. for extra logging, running validation set, virtual batch normalization etc.
        prefunc(x_mean)

        # Sample individuals (updates X)
        sample_population!(X, Z, x_mean, hyp_par.std, B)

        # Evaluate fitness function and sort in ascending order
        for k = 1:half_population_size*2
            xviews[k] = view(X, :, k) #evaluate fitness function
        end
        f_vals = pmap(f, workerpool, xviews)
        nevals += 2*half_population_size
        costlog[iter] = mean(f_vals)
        bestpop_index = sortperm(f_vals)
        Zsorted .= Z[:, bestpop_index]

        # Check if there is a new best individual
        bestfitness_current = f_vals[bestpop_index[1]]
        if bestfitness_current < bestfitness_ever #overwrite current best
            best_x .= X[:, bestpop_index[1]]
            bestfitness_ever = bestfitness_current
        end

        # Update evolution path
        sumUZ = Zsorted*uRank
        psigma .= (one(T) - csigma) .* psigma .+ sqrt( csigma * (2 - csigma) * mueff ) .* sumUZ
        rate::T = norm(psigma) / base

        # Calculate the gradients with the detection of the center's movement
        # and adapt the learning rates by the identification of the search phases
        fill!(GM, zero(T))
        if rate >= one(T)
            expZ = exp.(alpha * sqrt.(vec(sum(abs2, Zsorted; dims = 1))))
            uDist = (weights .* expZ) ./ dot(weights, expZ) .- 1/(half_population_size*2)
            Gdelta = Zsorted*uDist
            GM .= (uDist' .* Zsorted) * Zsorted' - sum(uDist) * LinearAlgebra.I
            etaB = etaBset[1]
            etaSigma = etaSigmaset[1]
        elseif rate < one(T)
            Gdelta = sumUZ
            GM .= (uRank' .* Zsorted) * Zsorted' - sum(uRank) * LinearAlgebra.I
            etaB = etaBset[2]
            if rate > T(0.1)
                etaSigma = etaSigmaset[2]
            else
                etaSigma = etaSigmaset[3]
            end
        end

        # Update the parameters
        x_mean .+= etaMu * T(hyp_par.std)* B * Gdelta
        Gsigma = tr(GM)/length_x

        # This is actually called GB in the paper and matlab, but we reuse GM as GM will not be used anymore
        GM .-= Gsigma * Diagonal(LinearAlgebra.I, length_x)
        hyp_par.std *= exp(etaSigma./2 .* Gsigma)
        B = B * exp(etaB/2 * GB) # Actually GB as mentionned above


        # Check for stopping criteria and print information
        if iter % hyp_par.printiter == 0
            @printf("Iter %i: Mean: %.5e \tStd: %.6f\tBest_f: %.5e\n", iter, costlog[iter], hyp_par.std, bestfitness_ever)
        end
        if bestfitness_ever<hyp_par.f_tol
            println("Stopping because function value below tolerance")
            break
        end
        if hyp_par.std < hyp_par.std_tol
            println("Stopping because std below std_tol")
            break
        end
        #=if norm(grad, 2)<hyp_par.dx_tol
            println("Stopping because gradient norm below tolerance")
            break
        end=#
        if nevals > hyp_par.maxevals
            println("Stopping because maximum allowed number of function evaluations reached")
            break
        end
    end

    return best_x, costlog

end


function sample_population!(X, Z, x_mean::Vector{T}, std, B) where {T}
    # Makes code more readable
    n = size(X, 1)
    Npop = size(X, 2)
    Nhalfpop = Int(Npop/2)

    # Sample from the distribution
    temp = randn(T, n, Nhalfpop)
    Z[:, 1:Nhalfpop] .= temp
    Z[:, Nhalfpop+1:end] .= -one(T) .* temp

    X .= T(std) .* (B*Z) .+ x_mean
    return nothing
end
