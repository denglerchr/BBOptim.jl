module BBOptim

using Distributed, Random, LinearAlgebra, Statistics, Printf
import Base.show

include("optimizer.jl")
export GradDesc, Momentum, Adam, GradDescAdaptive

include("es_hyp.jl")
include("es_opt.jl")
export EvolStrat

include("dxnes_hyp.jl")
include("dxnes_opt.jl")
export Dxnes

include("cmaes_hyp.jl")
include("cmaes_opt.jl")
export CMAES
export minimize

end
