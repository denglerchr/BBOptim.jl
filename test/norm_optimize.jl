@everywhere using BBOptim, LinearAlgebra
#@everywhere using BBOptim, Distributed, LinearAlgebra

# Function and initial guess

# Simple test function
@everywhere f(x) = norm([i*x[i] for i = 1:length(x)])
x0 = ones(Float32, 100)

#= Rosenbrock function
@everywhere f(x) = sum([100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2  for i = 1:(length(x)-1)])
x0 = zeros(Float32, 100)
=#

#= Styblinski–Tang function
@everywhere f(x) = 0.5*sum([x[i]^4 - 16*x[i]^2 + 5*x[i]  for i = 1:length(x)])
x0 = ones(Float32, 100)
=#

############################## Evol Strat #########################
# Settings for the Optimization. Get help with ?EsHyperparam
es = EvolStrat(x0)
es.maxiter = 3000
es.std = 3.0
es.population_size = 10
 # Momentum(0.03, 0.9, length(x0))
 # GradDesc(0.005)
 # GradDescAdaptive(0.005, 0.8)
 # Adam(x0; lr = 0.01)
es.optimizer = GradDescAdaptive(0.005, 0.8)
es.df_clip = 5.0
es.npersist = 3
es.printiter = 40

# Run opimization
@time xopt_es, costlog_es = minimize(f, x0, es)

############################## Dxnes #########################

dxnes = Dxnes(x0)
dxnes.maxiter = 3000
dxnes.std = 3.0
dxnes.population_size = 10
dxnes.printiter = 40

@time xopt_dxnes, costlog_dxnes = minimize(f, x0, dxnes)

############################## CMA ES #########################

cma_es = CMAES(x0)
cma_es.maxiter = 3000
cma_es.std = 3.0
cma_es.population_size = 10
cma_es.printiter = 40

# Run opimization
@time xopt_cmaes, costlog_cmaes = minimize(f, x0, cma_es)
