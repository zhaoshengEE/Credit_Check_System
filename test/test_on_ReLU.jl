using Test 

function ReLU(x::Float64)
    result = max(0.0, x)
    return result
end

@test ReLU(1.0) == 1.0
@test ReLU(100.0) == 100.0

@test ReLU(0.0) == 0.0

@test ReLU(-1.0) == 0.0
@test ReLU(-10.0) == 0.0
