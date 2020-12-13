using DataFrames, CSV # For loading data
using Plots # For plotting the total error in training process
using Distributions # For generating weights

argA = 0 # An argument value in binary
argB = 1 # Another arugment value in binary
bias = 1 # The bias term is used to help the neural network fit best for the input data

inputNum = 5  # Include bias term
hiddenNum = 4 # Include bias term
outputNum = 1
momentum = 0.9 # A term used to increase the neural network's tranining speed
learningRate = 0.2

###  Activation Function (Sigmoid)  ###
function sigmoid(x::Float64)
    result = (argB - argA) / (1 + exp(-x)) + argA
    return result
end

###  Activation Function (ReLU)  ###
function ReLu(x::Float64)
    result = max(0.0, x)
    return result
end

###  Training Function  ###
function train(input::Array{Float64,1}, expectedOutput::Float64)
    trainOutput = outputFor(input) # forward propagation
    updateWeight(trainOutput, expectedOutput) # backpropagation
    errorRate = 0.5 * (trainOutput - expectedOutput)^2
    return errorRate
end

###  Forward Propagation Function  ###
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
    outputNeuron[1] = sigmoid(outputNeuron[1])
    return outputNeuron[1]
end

###  Backpropagation Function  ###
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

###  Initilize Neurons and Weights  ###
inputNeuron = zeros(inputNum)
hiddenNeuron = zeros(hiddenNum)
outputNeuron = zeros(outputNum)

inputWeights = rand(Uniform(-0.5, 0.5), (inputNum, hiddenNum - 1))
hiddenWeights = rand(Uniform(-0.5, 0.5), (hiddenNum, outputNum))

### Error Signal for Neurons and Weights (Used in Backpropagation)  ###
inputWeightsDelta = zeros(inputNum, hiddenNum - 1)
hiddenWeightsDelta = zeros(hiddenNum, outputNum)

deltaHidden = zeros(hiddenNum)
deltaOutput = zeros(outputNum)

epoch = 0
errorRate = 0.0
acceptError = 0.05 # The threshold value of error rate

###  Read Data from the CSV File  ###
df = DataFrame(CSV.File("data/germandata.csv"))
head(df)

inputATrain = df[1:600,1]  # Credit history
inputBTrain = df[1:600,2]  # Other installment plans
inputCTrain = df[1:600,3]  # Property
inputDTrain = df[1:600,4]  # Telephone
expectedOutput = df[1:600,5] # Expected training output


epochVector = zeros(0) # Store epoch number for plotting
errorRateVector = zeros(0) # Store error rate in each epoch for plotting

###  Training Process  ###

while(epoch == 0 || errorRate > acceptError)
    errorRate = 0.0
    for index in 1:600
        input = convert(Array{Float64,1}, [inputATrain[index], inputBTrain[index], inputCTrain[index], inputDTrain[index]])
        output = convert(Float64, expectedOutput[index])
        errorRate += train(input, output)
        end
    println("Error at epoch $epoch is $errorRate")
    append!(epochVector, convert(Float64,epoch))
    append!(errorRateVector, errorRate)
    epoch += 1
end

### Plot the Training Process  ###
plot(epochVector, errorRateVector, title = "Total Squared Error in Training Process", lw = 3, legend = false)
xlabel!("Epoch")
ylabel!("Error")

###  Testing Process  ###
inputATest = df[601:1000,1]  # Credit history
inputBTest = df[601:1000,2]  # Other installment plans
inputCTest = df[601:1000,3]  # Property
inputDTest = df[601:1000,4]  # Telephone
expectedOutputTest = df[601:1000,5] # Expected training output

count = 0

for index in 1:400
    testInput = convert(Array{Float64,1}, [inputATest[index], inputBTest[index], inputCTest[index], inputDTest[index]])
    testOutput = outputFor(testInput)
    realOutput = convert(Float64, expectedOutputTest[index])

    # The trail can be considered as successful only if the testing value is 
    # within the accepted error range
    if testOutput > realOutput - 0.05 && testOutput < realOutput + 0.05
        count += 1
    end

end

### Calculate the Accuracy  ###
accuracy = (count / 400) * 100
println("The accuracy of this trained neural netowrk is $accuracy%")
