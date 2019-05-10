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

@inline function minimize!(params::AbstractVector, grad::AbstractVector, opt::GradDesc, f::Function)
    params .-= opt.lr.*grad
end

function show(io::IO, opt::GradDesc)
    println("Gradient Descend:")
    println("\tLearning rate: $(opt.lr)")
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

@inline function minimize!(params::AbstractVector, grad::AbstractVector, opt::Momentum, f::Function)
    opt.v .= opt.mom.* opt.v .+ opt.lr.*grad # Accumulate momentum
    params .-= opt.v # Apply momentum
end

function show(io::IO, opt::Momentum)
    println("Gradient Descend with Momentum:")
    println("\tLearning rate: $(opt.lr)")
    println("\tMomentum: $(opt.mom)")
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

@inline function minimize!(params::AbstractVector, grad::AbstractVector, opt::Adam, f::Function)
    # Update betat (contains beta1,2^t)
    opt.betat[1] *= opt.beta1
    opt.betat[2] *= opt.beta2

    # Update running mean and variance
    opt.mt .= opt.mt .* opt.beta1 .+ (1.0 - opt.beta1) .* grad
    opt.vt .= opt.vt .* opt.beta2 .+ (1.0 - opt.beta2) .* (grad.^2)

    # Update parameters
    params .-= opt.lr .* ( (opt.mt ./ (1.0-opt.betat[1]) ) ./ (sqrt.(opt.vt ./ (1.0-opt.betat[2]) ) .+ opt.epsilon) )
end

function show(io::IO, opt::Adam)
    println("Adam:")
    println("\tLearning rate: $(opt.lr)")
    println("\tbeta1: $(opt.beta1)")
    println("\tbeta2: $(opt.beta2)")
    println("\tepsilon: $(opt.epsilon)")
end

###################### GradDesc with adaptive learning rate ####################

mutable struct GradDescAdaptive<:AbstractOptimizer
    lr::Number # Learning Rate
    adaptstep::Number

    # Check if learning rate >0
    function GradDescAdaptive(lr, adaptstep)
        @assert lr > 0.0
        @assert adaptstep > 0.0
        return new(lr, adaptstep)
    end
end

@inline function minimize!(params::AbstractVector, grad::AbstractVector, opt::GradDescAdaptive, f::Function)
    # Line Search
    F1 = f(params-opt.lr/opt.adaptstep*grad)
    F2 = f(params-opt.lr*opt.adaptstep*grad)
    F1>F2 ? opt.lr = opt.lr*opt.adaptstep : opt.lr = opt.lr/opt.adaptstep

    params .-= opt.lr.*grad
end

function show(io::IO, opt::GradDescAdaptive)
    println("Gradient Descend with adaptive learning rate:")
    println("\tLearning rate: $(opt.lr)")
    println("\tAdaption rate: $(opt.adaptstep)")
end
