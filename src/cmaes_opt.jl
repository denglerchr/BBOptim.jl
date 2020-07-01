function minimize(f::Function, x0::Vector{T}, hyp_par::CMAES; prefunc::Function = x->nothing, workerpool::AbstractWorkerPool = default_worker_pool()) where {T}

    # Preallocate variables
    xmean = copy(x0) #objective variables initial point
    costlog = zeros(hyp_par.maxiter)
    N = length(x0)

    #Strategy parameter setting: Selection
    mu::Int = floor(Int, hyp_par.population_size/2)
    weights = log(hyp_par.population_size/2 + T(0.5)) .- log.(1:T(hyp_par.population_size/2)) #array for weighted recombination
    normalize!(weights, 1) #normalize weights
    mueff = one(T)/sum(x->x^2, weights)

    #Strategy parameter setting: Adaptation
    cc::T = (4 + mueff/N) / (N + 4 + 2*mueff/N) #time constant for cumulation of the covariance matrix C
    cs::T = (mueff + 2) / (N + mueff + 5) #time constant for cumulation for variance control
    c1::T = 2 / ((N + 1.3)^2 + mueff) #learning rate for rank-one update of C
    cmu::T = min(1 - c1, 2 * (mueff- 2 + 1/mueff) / ((N + 2)^2 + mueff)) # and for rank-mu update
    damps::T = 1 + 2*max(0, sqrt((mueff - 1 ) / (N + 1)) - 1) + cs #damping for sigma

    #Initialize dynamic (internal) strategy parameters and constants
    pc = zeros(T, N) #evolution paths for C and sigma
    ps = zeros(T, N)

    B = I #defines the coordinate system
    D = ones(T, N) #defines the scaling
    C_full = Matrix{T}(I, N, N) #covariance matrix C
    C = Symmetric(C_full)
    invsqrtC = I #C^-0.5
    xold = similar(x0) #preallocate
    best_x = similar(x0); bestfitness_ever = Inf
    eigenval = 0 #track update of B and D
    chiN::T = N^0.5*(1 - 1/(4*N) + 1/(21*N^2)) #expectation of ||N(0,I)|| == norm(randn(N,1))

    #Generation Loop
    counteval = 0
    arx = zeros(T, N, hyp_par.population_size)
    artemp = zeros(T, N, mu+1)
    arxvec = Vector{AbstractVector}(undef, hyp_par.population_size)

    for iter = 1:hyp_par.maxiter     #counteval < stopeval

        # Check validation error
        prefunc(best_x)

        #Generate and evaluate lamda offspring
        arx .= xmean .+ hyp_par.std*B*(D .* randn(T, N, hyp_par.population_size))
        for k = 1:hyp_par.population_size
            arxvec[k] = view(arx, :, k) #evaluate fitness function
        end
        arfitness = pmap(f, workerpool, arxvec)
        counteval += hyp_par.population_size

        #Sort by fitness and compute weighted mean into xmean
        arindex = sortperm(arfitness) #minimization
        xold .= xmean
        xmean .= arx[:, arindex[1:mu]] * weights #recombination, new mean value
        bestfitness_current = minimum(arfitness)
        if bestfitness_current < bestfitness_ever #overwrite current best
            best_x .= arx[:, arindex[1]]
            bestfitness_ever = bestfitness_current
        end

        #Cumulation: Update evolution paths
        ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / hyp_par.std
        hsig = sum(x->x^2, ps) / (1 - (1 - cs)^(2 * counteval/hyp_par.population_size))/N < 2 + 4/(N + 1) # either 1 or 0

        #Adapt covariance matrix C
        pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / hyp_par.std
        artemp[:, 1] .= sqrt(c1).*pc
        @inbounds artemp[:, 2:mu+1] .= sqrt(cmu/hyp_par.std).*arx[:, arindex[1:mu]].*sqrt.(weights)'
        cscale::T = (1 - c1 - cmu + c1*(1 - hsig)*cc*(2 - cc))
        BLAS.syrk!('U', 'N', one(T), artemp, cscale, C_full)

        #Adapt step size sigma
        hyp_par.std = hyp_par.std * exp((cs/damps) * (norm(ps)/chiN - 1))

        #Update B and D from C
        if counteval - eigenval > hyp_par.population_size/(c1 + cmu)/N/10 #to achieve O(N^2)
            eigenval = counteval
            D, B = eigen(C) #eigen decomposition
            D = sqrt.( max.(eps(T), D) ) #D contains standard deviations now
            invsqrtC = B * Diagonal(1 ./ D) * B'
        end
        cost_new = mean(arfitness)
        costlog[iter] = cost_new

        #Break, if fitness is good enough
        if bestfitness_current <= hyp_par.f_tol
            println("Converged after $iter iterations with final costs: $cost_new")
            break
        elseif counteval > hyp_par.maxevals || maximum(D) > 1e7 * minimum(D)
            println("Stopped after $iter iterations because of bad conditioning")
			break
        end
        if hyp_par.std < hyp_par.std_tol
            println("Stopping because std below std_tol")
            break
        end
        if iter % hyp_par.printiter == 0
            @printf("Iter %i: Mean: %.5e \tStd: %.6f\tBest_f: %.5e\n", iter, cost_new, hyp_par.std, bestfitness_ever)
        end

    end
    return best_x, costlog
end
