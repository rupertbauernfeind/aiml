import math
import random
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self):
        self.__layers = []
        self.__inputLayer = None
        self.__outputLayer = None

    def addLayer(self, size, activationFunction):
        """
        Adds a new Layer to your Neural Network.
        The first Layer is the InputLayer and the last one is the Output.
        :param size: Number of Neuron added to the Layer
        :param activationFunction: static Neuron-Function. i.e. tanh()
        """
        anteriorLayer = self.__outputLayer
        newLayer = []

        # create Neurons and append to the Layer
        for i in range(0, size):
            n = Neuron(anteriorLayer, activationFunction)
            newLayer.append(n)

        self.__layers.append(newLayer)
        self.__outputLayer = newLayer
        self.__inputLayer = self.__layers[0]

        # add posteriorLayers to Neurons
        if len(self.__layers) > 1:
            for n in anteriorLayer:
                if isinstance(n, Neuron):
                    n.setPosteriorLayer(newLayer)

    def forwardPropagation(self, inputList):
        """
        Pass an Input through the Neural Network and get the final Output as a list
        :param inputList: Input-List, must have the size like the Input-Layer
        :return: Output Layer Value
        """
        if len(self.__layers) < 2:
            raise NameError("Not enough Layer to forwardPass. Minimum Input and Output Layer!")

        if len(inputList) != len(self.__inputLayer):
            raise NameError("Number of inputs doesn't match with input Layer")

        # set the Output of the Input Layer
        for i in range(0, len(self.__inputLayer)):
            n = self.__inputLayer[i]
            n.setOut(inputList[i])

        # pass forward through the hidden Layer to the Output Layer
        return [n.getOut() for n in self.__outputLayer]

    def getMeanSquareError(self, labels):
        mse = 0
        for i in range(0, len(self.__outputLayer)):
            n = self.__outputLayer[i]
            mse += 0.5 * ((labels[i] - n.getOut()) ** 2)
        return mse

    def backwardPropagation(self, learningRate, labels):
        if len(self.__layers) < 2:
            raise NameError("Not enough Layer to forwardPass. Minimum Input and Output Layer!")

        if len(labels) != len(self.__outputLayer):
            raise NameError("Not enough Labels for the Number of output Neurons")

        # update the output Layers Weight
        for i in range(0, len(self.__outputLayer)):
            n = self.__outputLayer[i]
            n.updateWeight(learningRate, labels[i])

        # update the hidden Layers Weight
        self.__layers.reverse()
        for layer in self.__layers:
            if layer == self.__outputLayer:
                continue
            if layer == self.__inputLayer:
                continue
            for n in layer:
                n.updateWeight(learningRate)
        self.__layers.reverse()

    def getWeightPattern(self):
        backup = []
        for layer in self.__layers:
            for n in layer:
                if isinstance(n, Neuron):
                    backup.append(n.getWeights)
        return backup

    def setWeightPattern(self, backup):
        i = 0
        for layer in self.__layers:
            for n in layer:
                if isinstance(n, Neuron):
                    n.setWeights(backup[i])
                    i += 1


class Neuron:

    def __init__(self, anteriorLayer, activationFunction):
        """
        Create a new Neuron with the linked anterior Layer and the activation-function
        :param anteriorLayer: Layer before the Neurons Layer
        :param activationFunction: a function with one parameter, called for evaluating the Output
        """
        self.__out = None
        self.__net = None
        self.__delta = None
        self.__posteriorLayer = None
        self.__activationFunction = activationFunction
        self.__anteriorLayer = anteriorLayer
        self.__weights = []
        if not self.__isInputLayer():
            # Add weights in size of anterior Layer plus Bias-Weight
            for i in range(0, len(self.__anteriorLayer) + 1):
                self.__weights.append(Neuron.randomWeight())

    def setPosteriorLayer(self, posteriorLayer):
        self.__posteriorLayer = posteriorLayer

    def setOut(self, out):
        self.__out = out

    def setWeights(self, weights):
        self.__weights = weights

    def getNet(self):
        if not self.__isInputLayer():
            self.__net = 0
            for i in range(0, len(self.__anteriorLayer)):
                self.__net += self.__anteriorLayer[i].getOut() * self.__weights[i]
            # add Bias Weight
            self.__net += self.__weights[-1]
        return self.__net

    def getOut(self):
        if not self.__isInputLayer():
            self.__out = self.__activationFunction(self.getNet())
        return self.__out

    def getWeights(self, index=-1):
        if index == -1:
            return self.__weights.copy()
        else:
            if index < 0 or index >= len(self.__weights):
                raise NameError("Index in weights is out of Range")
            return self.__weights[index]

    def getDelta(self):
        if self.__isInputLayer():
            raise NameError("InputLayer is not able to contain a delta Value")
        return self.__delta

    def updateWeight(self, learningRate=0.001, label=None):
        """
        Calculate the delta value for weight update: delta = f'(net) * (true - out)
        :param learningRate:
        :param label: the should-Value if the Node is in the Output-Layer
        """
        if self.__isInputLayer():
            raise NameError("Weight update for InputLayer is not allowed")

        # compute the delta
        if self.__isOutputLayer():
            if label is None:
                raise NameError("updateDelta needs a Should-Output-Value for training the Network")
            self.__delta = self.__activationFunction(self.getNet(), True) * (label - self.getOut())
        else:  # Hidden Layer
            posteriorDeltaSum = 0
            for i in range(0, len(self.__posteriorLayer)):
                n = self.__posteriorLayer[i]
                posteriorDeltaSum += n.getDelta() * n.getWeights(i)
            self.__delta = self.__activationFunction(self.getNet(), True) * posteriorDeltaSum

        # update the weights
        for i in range(0, len(self.__weights) - 1):
            n = self.__anteriorLayer[i]
            deltaWeight = learningRate * self.__delta * n.getOut()
            self.__weights[i] += deltaWeight
        # bias
        deltaWeight = learningRate * self.__delta * 1
        self.__weights[-1] += deltaWeight

    def __isInputLayer(self):
        return self.__anteriorLayer is None or len(self.__anteriorLayer) == 0

    def __isOutputLayer(self):
        return self.__posteriorLayer is None or len(self.__posteriorLayer) == 0

    @staticmethod
    def identity(x, derivative=False):
        if derivative:
            return 1
        else:
            return x

    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - (math.tanh(x) * math.tanh(x))
        else:
            return math.tanh(x)

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            f = 1 / (1 + (math.e ** (-x)))
            return f * (1 - f)
        else:
            return 1 / (1 + (math.e ** (-x)))

    @staticmethod
    def randomWeight():
        return random.randrange(-100, 100, 1) / 100000


def loadInputData():
    trainingData = []
    executionData = []

    inputStrings = []

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

    trainingSet = trainingData[:int(len(trainingData) * 0.7)]
    validationSet = trainingData[int(len(trainingData) * 0.7):]

    return trainingSet, validationSet, executionData


def main():
    nn = NeuralNetwork()
    nn.addLayer(2, None)
    nn.addLayer(3, Neuron.tanh)
    nn.addLayer(1, Neuron.tanh)

    trainingSet, validationSet, executionData = loadInputData()

    negativeX = []
    negativeY = []
    positiveX = []
    positiveY = []
    for data in trainingSet:
        if data[2] == -1:
            negativeX.append(data[0])
            negativeY.append(data[1])
        else:
            positiveX.append(data[0])
            positiveY.append(data[1])

    for data in validationSet:
        if data[2] == -1:
            negativeX.append(data[0])
            negativeY.append(data[1])
        else:
            positiveX.append(data[0])
            positiveY.append(data[1])

    plt.plot(negativeX, negativeY, "o", label="line 1")
    plt.plot(positiveX, positiveY, "o", label="line 2")

    for i in range(0, 1000):

        for data in trainingSet:
            nn.forwardPropagation(data[:2])
            nn.backwardPropagation(0.01, data[2:])

        for data in validationSet:
            nn.forwardPropagation(data[:2])
            nn.backwardPropagation(0.01, data[2:])

    negX = []
    negY = []
    posX = []
    posY = []
    #executionData = []
    # for i in range(-100, 100):
    #     for j in range(-100, 100):
    #         executionData.append([i / 100, j / 100])

    for data in executionData:
        trueOutput = nn.forwardPropagation(data[:2])[0]
        if trueOutput > 0:
            posX.append(data[0])
            posY.append(data[1])
            print("+1")
        else:
            negX.append(data[0])
            negY.append(data[1])
            print("-1")

    plt.plot(negX, negY, "o", label="line 3")
    plt.plot(posX, posY, "o", label="line 4")
    plt.show()


if __name__ == "__main__":
    main()
