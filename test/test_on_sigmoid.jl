using Test

argA = 0
argB = 1
function sigmoid(x::Float64)
    result = (argB - argA) / (1 + exp(-x)) + argA
    return result
end

@test sigmoid(0.0) ≈ 0.5
@test sigmoid(1.0) ≈ 0.7310585786300049
@test sigmoid(2.0) ≈ 0.8807970779778823
@test sigmoid(-1.0) ≈ 0.2689414213699951
@test sigmoid(-2.0) ≈ 0.11920292202211755


