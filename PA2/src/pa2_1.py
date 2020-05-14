import math
import random


class Neuron:

    # Initialize the Neuron for with random weights
    def __init__(self, inputCount=2, learningRate=0.001):

        # Stores the learningRate given as parameter
        self.learningRate = learningRate

        # random weight-initialisation
        self.weights = []
        for i in range(0, inputCount + 1):
            self.weights.append(Neuron.randomWeight())

    def output(self, inputVector):
        # Add bias Input
        inputs = inputVector.copy()
        inputs.insert(0, 1)

        # Check, if input-Vector is as big as the weights-vector minus 1
        if len(inputs) != len(self.weights):
            raise NameError("Incorrect Size in Input Layer")

        # compute the Input-Sum
        net = 0
        for i in range(0, len(inputs)):
            net = net + inputs[i] * self.weights[i]

        output = Neuron.tanh(net)

        return output, net, inputs

    def train(self, inputVector, label):
        # compute the Output
        output, net, inputs = self.output(inputVector)
        # compute the weight-Change deltaWeight for each weight
        delta = self.learningRate * Neuron.tanhDerivative(net) * (label - output)
        for i in range(0, len(self.weights)):
            deltaWeight = delta * inputs[i]
            self.weights[i] = self.weights[i] + deltaWeight

    @staticmethod
    def randomWeight():
        return random.randrange(-100, 100, 1) / 100000

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def tanhDerivative(x):
        n = Neuron.tanh(x)
        return 1 - (n * n)


def loadInputData():
    trainingData = []
    executionData = []

    inputStrings = []

    # store the Input Values
    while True:
        try:
            inputStrings.append(input())
        except EOFError:
            break
    # sort the Data in training and execution Data
    for s in inputStrings:
        valueStrings = s.split(',')
        values = [float(valueStrings[0]), float(valueStrings[1])]
        if len(valueStrings) > 2:
            values.append(float(valueStrings[2]))
            if values[0] == 0 and values[1] == 0 and values[2] == 0:
                continue
            trainingData.append(values)
        else:
            executionData.append(values)

    # normalize the Data with max Magnitude per Input
    maxMagnitudeX0 = 0
    maxMagnitudeX1 = 0
    for data in trainingData:
        if abs(float(data[0])) > maxMagnitudeX0:
            maxMagnitudeX0 = abs(float(data[0]))
        if abs(float(data[1])) > maxMagnitudeX1:
            maxMagnitudeX1 = abs(float(data[1]))

    for data in executionData:
        if abs(float(data[0])) > maxMagnitudeX0:
            maxMagnitudeX0 = abs(float(data[0]))
        if abs(float(data[1])) > maxMagnitudeX1:
            maxMagnitudeX1 = abs(float(data[1]))

    for data in trainingData:
        data[0] = data[0] / maxMagnitudeX0
        data[1] = data[1] / maxMagnitudeX1

    for data in executionData:
        data[0] = data[0] / maxMagnitudeX0
        data[1] = data[1] / maxMagnitudeX1

    return trainingData, executionData


def trainNetwork(neuron, trainingData, maxDepth):
    for i in range(0, maxDepth):
        for values in trainingData:
            inputs = values[:2]
            label = values[2]
            neuron.train(inputs, label)


def executeNetwork(neuron, executionData):
    for inputs in executionData:
        output, net, inputsReturn = neuron.output(inputs)
        if output > 0:
            print("+1")
        else:
            print("-1")


def main():
    neuron = Neuron(2, 0.01)
    trainingData, executionData = loadInputData()
    trainNetwork(neuron, trainingData, 10000)
    executeNetwork(neuron, executionData)


if __name__ == "__main__":
    main()
