using DataFrames, CSV # For loading data
using Plots
using Distributions # For generating weights

argA = 0
argB = 1
bias = 1

inputNum = 5  # Include bias term
hiddenNum = 6 # Include bias term
outputNum = 1
momentum = 0.9
learningRate = 0.1

function sigmoid(x::Float64)
    return ((argB - argA) / (1 + exp(-x)) + argA)
end

function train(input::Array{Float64,1}, expectedOutput::Float64)
    trainOutput = outputFor(input) # forward propagation
    println(trainOutput)
    updateWeight(trainOutput, expectedOutput) # backpropagation
    errorRate = 0.5 * (trainOutput - expectedOutput)^2
    # println(errorRate)
    return errorRate
end

function outputFor(input::Array{Float64,1})
    for i = 1:inputNum - 1
        inputNeuron[i] = input[i]
    end
    
    inputNeuron[inputNum] = bias
    hiddenNeuron[hiddenNum] = bias
    
    for i = 1:hiddenNum - 1
        for j = 1:inputNum
           hiddenNeuron[i] += inputWeights[j, i] * inputNeuron[j]
        end
        hiddenNeuron[i] = sigmoid(hiddenNeuron[i])
    end
    
    for k = 1:hiddenNum
        outputNeuron[1] += hiddenWeights[k] * hiddenNeuron[k]
    end
    
    return outputNeuron[1]
end

function updateWeight(trainOutput::Float64, expectedOutput::Float64)
    deltaOutput[1] = trainOutput * (1 - trainOutput) * (expectedOutput - trainOutput)

    for k = 1:hiddenNum
        hiddenWeights[k] +=  (momentum * hiddenWeightsDelta[k]) + (learningRate * deltaOutput[1] * hiddenNeuron[k]);
        hiddenWeightsDelta[k] = (momentum * hiddenWeightsDelta[k]) + (learningRate * deltaOutput[1] * hiddenNeuron[k]);
    end
    
    for k = 1:hiddenNum - 1
        sum = 0.0
        deltaHidden[k] = hiddenNeuron[k] * (1 - hiddenNeuron[k])
        sum += deltaOutput[1] * hiddenWeights[k]
        deltaHidden[k] *= sum
    end
    
    for j = 1:hiddenNum - 1
        for i = 1:inputNum
            inputWeights[i, j] += (momentum * inputWeightsDelta[i, j]) + (learningRate * deltaHidden[j] * inputNeuron[i])
            inputWeightsDelta[i, j] = (momentum * inputWeightsDelta[i, j]) + (learningRate * deltaHidden[j] * inputNeuron[i])
        end
    end
end

inputNeuron = zeros(inputNum)
hiddenNeuron = zeros(hiddenNum)
outputNeuron = zeros(outputNum)

inputWeights = rand(Uniform(-0.5, 0.5), (inputNum, hiddenNum - 1))
hiddenWeights = rand(Uniform(-0.5, 0.5), (hiddenNum, outputNum))

inputWeightsDelta = zeros(inputNum, hiddenNum - 1)
hiddenWeightsDelta = zeros(hiddenNum, outputNum)

deltaHidden = zeros(hiddenNum)
deltaOutput = zeros(outputNum)

epoch = 0
errorRate = 0.0
acceptError = 0.05

df = DataFrame(CSV.File("data/trainingdata.csv"))
head(df)

inputATrain = df[1:16,1]  # Credit history
inputBTrain = df[1:16,2]  # Other installment plans
inputCTrain = df[1:16,3]  # Property
inputDTrain = df[1:16,4]  # Telephone
expectedOutput = df[1:16,5] # Expected training output

while(epoch == 0 || errorRate > acceptError)
    errorRate = 0.0
    for index in 1:16
        input = convert(Array{Float64,1}, [inputATrain[index], inputBTrain[index], inputCTrain[index], inputDTrain[index]])
        # print(input)
        output = convert(Float64, expectedOutput[index])
        # print(output)
        errorRate += train(input, output)
        # println(errorRate)
        end
    print(epoch)
    # print(errorRate)
    epoch += 1
end