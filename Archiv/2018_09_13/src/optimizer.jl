abstract type AbstractOptimizer end

###################### (Stochastic) Gradient Descend ####################
struct GradDesc<:AbstractOptimizer
    lr::Number # Learning Rate

    # Check if learning rate >0
    function GradDesc(lr)
        @assert lr>0.0
        return new(lr)
    end
end

function optimize!(params::AbstractVector, grad::AbstractVector, opt::GradDesc)
    params .-= opt.lr.*grad
end

###################### Gradient Descend with Momentum ####################
struct Momentum<:AbstractOptimizer
    lr::Number      # Learing Rate
    mom::Number     # Momentum parameter

    v::Vector       # Accumulated Momentum

    # Check if inputs make sense
    function Momentum(lr::Number, mom::Number, v::Vector)
        @assert lr>0.0
        @assert 0.0<=mom<1.0
        return new(lr, mom, v)
    end
end

# Alternative Constructors
Momentum(lr::Number, mom::Number, nparams::Number) = Momentum(lr, mom, zeros(nparams))

function optimize!(params::AbstractVector, grad::AbstractVector, opt::Momentum)
    opt.v .= opt.mom.* opt.v .+ opt.lr.*grad # Accumulate momentum
    params .-= opt.v # Apply momentum
end

###################### Adam Update ####################
struct Adam<:AbstractOptimizer
    lr::Number
    beta1::Number
    beta2::Number
    epsilon::Number
    mt::Vector
    vt::Vector
    betat::Vector

    function Adam(lr, beta1, beta2, epsilon, mt, vt)
        @assert lr > 0.0
        @assert 0.0 <= beta1 < 1.0
        @assert 0.0 <= beta2 < 1.0
        @assert epsilon > 0.0
        return new(lr, beta1, beta2, epsilon, mt, vt, ones(2))
    end
end

Adam(params::Vector ;lr::Number = 0.001, beta1::Number = 0.9, beta2::Number = 0.999, epsilon::Number = 1e-8) = Adam(lr, beta1, beta2, epsilon, zeros(length(params)), zeros(length(params)))

function optimize!(params::AbstractVector, grad::AbstractVector, opt::Adam)
    # Update betat (contains beta1,2^t)
    opt.betat[1] *= opt.beta1
    opt.betat[2] *= opt.beta2

    # Update running mean and variance
    opt.mt .= opt.mt .* opt.beta1 .+ (1.0 - opt.beta1) .* grad
    opt.vt .= opt.vt .* opt.beta2 .+ (1.0 - opt.beta2) .* (grad.^2)

    # Update parameters
    params .-= opt.lr .* ( (opt.mt ./ (1.0-opt.betat[1]) ) ./ (sqrt.(opt.vt ./ (1.0-opt.betat[2]) ) .+ opt.epsilon) )
end
