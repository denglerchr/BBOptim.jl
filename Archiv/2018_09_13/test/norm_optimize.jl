using EvolStrat

# Function and initial guess
@everywhere f(x) = norm([i*x[i] for i = 1:length(x)])
x0 = ones(2000)

# Settings for the Optimization. Get help with ?EsHyperparam
settings = InitEsHyperparam(x0)
settings.maxiter = 200
settings.std = 1.0
settings.optimizer = Adam(x0; lr = 0.01) # Momentum(0.03, 0.9, length(x0)) # GradDesc(0.03) # Adam(x0; lr = 0.01)
settings.df_clip = 5.0

# Run opimization
xopt, costlog = es_opt(f, x0, settings)
