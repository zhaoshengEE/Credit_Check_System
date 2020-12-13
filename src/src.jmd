Let's load some packages that is necessary for this project first

```julia
using DataFrames, CSV # For loading data
using Plots # For plotting the total error in training process
using Distributions # For generating weights
```

Before diving into this project, I do some pre-possessing on the original dataset, which is publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
I select some inputs and convert all the data into binary format. The table below illustrates the four selected inputs and the output of the neural network, along with their meanings in the binary format.

|            |      0      |      1       |
|:----------: | :-----------: |:------------:|
| Credit history | Had delay in paying off credit | All credits are paid back |
| Other installment plans | Not have other installment plans | Have other installment plans |
| Property | Not have other properties | Have other properties |
| Telephone | Not have registered telephone number | Have registered telephone number |
| Output | Bad credit | Good credit |

Also in the neural network, I decide to add bias terms in input layer and hidden layer. The bias terms can help the neural network fit best for the input data.

The image below shows the structure of the neural network I am implementing in this project:
![neural_network.png](img/neural_network.png)

I also assign other two variables called learningRate and momentum resepcitvely. The leanringRate is the learning speed of the nerual network and should not be too large. A large learning rate can lead to oscillations during the training process.
In this case, I assign a relatively small value to the learningRate and assign a relatively large value to the momentum. The momentum is used to motivate the learning of the neural network.

```julia
argA = 0
argB = 1
bias = 1

inputNum = 5
hiddenNum = 4
outputNum = 1
momentum = 0.9
learningRate = 0.2
```

An activation function is another essential element in a neural network. Here I provide two types of activation functions, one is the sigmoid function, the other is the ReLU function.
In this project, I am implementing the sigmoid function as activation function. In the future development on this project, one can try using the ReLU function as activation function.

```julia
###  Activation Function (Sigmoid)  ###
function sigmoid(x::Float64)
    result = (argB - argA) / (1 + exp(-x)) + argA
    return result
end

###  Activation Function (ReLU)  ###
function ReLU(x::Float64)
    result = max(0.0, x)
    return result
end
```

Then I start writing the training function, which consists of forward propagation function (`outputFor` function) and backpropagation function (`updateWeight` function).
The `train` function is going to be called a bunch of times during the training process, it returns the error between the training output and the expected output.

```julia
function train(input::Array{Float64,1}, expectedOutput::Float64)
    trainOutput = outputFor(input) # forward propagation
    updateWeight(trainOutput, expectedOutput) # backpropagation
    errorRate = 0.5 * (trainOutput - expectedOutput)^2
    return errorRate
end
```

The following function is the forward propagation function. This function first loads all the input data into the neuron at input layer.
Then it assigns bias terms to the last elements in input layer and hidden layer resepcitvely. After that the function calcualtes the values in all the neurons in hidden layers and output layers.
The function use the sigmoid function as activation function to map the values in hidden neurons and output neuron to a specified range.
Eventually, the `outputFor` function returns the final value in output neuron.

```julia
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
```

The following function is the backpropagation function.
The `updateWeight` function computes the error signal in the output neuron. Then it calculates the error signal in the weights between the hidden layer and the output layer.
After that, the `updateWeight` function calculates the error signal in each hidden neuron, eventually it computes the error signal in the weights between the input layer and the hidden layer.

```julia
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
```

Next, I initialize all the neuron values with 0:

```julia
inputNeuron = zeros(inputNum)
hiddenNeuron = zeros(hiddenNum)
outputNeuron = zeros(outputNum)
```

I initialize all the weights with random values in the range between -0.5 and 0.5:

```julia
inputWeights = rand(Uniform(-0.5, 0.5), (inputNum, hiddenNum - 1))
hiddenWeights = rand(Uniform(-0.5, 0.5), (hiddenNum, outputNum))
```

After that, I initialize all the error signals with 0:

```julia
inputWeightsDelta = zeros(inputNum, hiddenNum - 1)
hiddenWeightsDelta = zeros(hiddenNum, outputNum)

deltaHidden = zeros(hiddenNum)
deltaOutput = zeros(outputNum)
```

and the `epoch`, `errorRate`, `acceptError`, in this project, I set the accept error as 5%

```julia
epoch = 0
errorRate = 0.0
acceptError = 0.05 
```

Next I load all the data in the CSV file into the dataframe variable `df`:

```julia
df = DataFrame(CSV.File("data/germandata.csv"))
head(df)
```

The `df` has 1000 sets of data. Here I decide to use the first 600 instances as training dataset, and use the other 400 instances as validation dataset (or say testing dataset):

```julia
###  Training Dataset  ###
inputATrain = df[1:600,1]  # Credit history
inputBTrain = df[1:600,2]  # Other installment plans
inputCTrain = df[1:600,3]  # Property
inputDTrain = df[1:600,4]  # Telephone
expectedOutput = df[1:600,5] # Expected training output

###  Testing Dataset  ###
inputATest = df[601:1000,1]  # Credit history
inputBTest = df[601:1000,2]  # Other installment plans
inputCTest = df[601:1000,3]  # Property
inputDTest = df[601:1000,4]  # Telephone
expectedOutputTest = df[601:1000,5] # Expected training output
```

and initialize two vectors to store epoch numbers and the error rate in each epoch. These two vectors are designed for the plot later.

```julia
epochVector = zeros(0) # Store epoch number for plotting
errorRateVector = zeros(0) # Store error rate in each epoch for plotting
```

Now, the training process starts, the neural network takes each set of data in the training set and sends the input set and the expected output to the `traing` function.
The `while` loop will not stop until the total error rate, got from the `train` function in a epoch, is not larger than the accept error rate (i.e. 0.05)

```julia
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
```

Then I plot the training process, the training process is shown by the `errorRate` in each epoch. As the training proceeds, the `errorRate` becomes smaller and smaller, which shows the neural network is leanring.

```julia
### Plot the Training Process  ###
plot(epochVector, errorRateVector, title = "Total Squared Error in Training Process", lw = 3, legend = false)
xlabel!("Epoch")
ylabel!("Error")
```

![total_squared_error.PNG](img/total_squared_error.PNG)

Here comes the validation process, I input the other 400 instances into the trained neural network one by one. If the value computed by the neural network is within the range of accepted error rate, then the corresponding test is considered as successful.

```julia
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
```

Finally, I calculate the success rate of the neural network and print the result on the screen:

```julia
accuracy = (count / 400) * 100
println("The accuracy of this trained neural netowrk is $accuracy%")
```

![test_result.PNG](img/test_result.PNG)
