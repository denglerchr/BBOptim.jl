@everywhere using BBOptim, LinearAlgebra
#@everywhere using BBOptim, Distributed, LinearAlgebra

# Function and initial guess
@everywhere f(x) = norm([i*x[i] for i = 1:length(x)])
x0 = ones(Float32, 100)


############################## Evol Strat #########################
# Settings for the Optimization. Get help with ?EsHyperparam
es = EvolStrat(x0)
es.maxiter = 3000
es.std = 1.0
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
@time xopt, costlog = minimize(f, x0, es)

############################## Dxnes #########################

dxnes = Dxnes(x0)
dxnes.maxiter = 3000
dxnes.population_size = 10
dxnes.printiter = 40

@time xopt, costlog = minimize(f, x0, dxnes)

############################## CMA ES #########################

cma_es = CMAES(x0)
cma_es.maxiter = 3000
cma_es.population_size = 10
cma_es.printiter = 40

# Run opimization
@time xopt, costlog = minimize(f, x0, cma_es)
