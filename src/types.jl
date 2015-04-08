
import Optim.OptimizationState
import Optim.OptimizationTrace
import Optim.DifferentiableFunction

immutable ProximalOperator
    g::Function
    prox_g!::Function
end
