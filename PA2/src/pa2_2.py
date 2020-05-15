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
            - stores the __inputLayer and __outputLayer
            - stores the posterior Layer for each Layer
        :param size: Number of Neuron added to the Layer
        :param activationFunction: static Neuron-Function. i.e. tanh()
        """
        anteriorLayer = self.__outputLayer  # None if its the first input Layer
        newLayer = []

        # create Neurons and append to the Layer
        for i in range(0, size):
            n = Neuron(anteriorLayer, activationFunction)
            newLayer.append(n)

        self.__layers.append(newLayer)
        self.__outputLayer = newLayer
        self.__inputLayer = self.__layers[0]

        # set the actual Layer to the anterior layer as posterior layer
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

        # update the hidden Layers Weight
        self.__layers.reverse()
        for layer in self.__layers:
            if layer == self.__outputLayer:
                # update the output Layers Weight
                for i in range(0, len(self.__outputLayer)):
                    n = self.__outputLayer[i]
                    n.updateWeight(learningRate, labels[i])
            elif layer == self.__inputLayer:
                continue
            else:
                for n in layer:
                    n.updateWeight(learningRate)
        self.__layers.reverse()

    '''
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
    '''

    def train(self, trainingsData, labelSize, learningRate):
        for data in trainingsData:
            self.forwardPropagation(data[:-labelSize])
            self.backwardPropagation(learningRate, data[-labelSize:])

    def validate(self, validationData, labelSize=1):
        mse = 0
        for data in validationData:
            self.forwardPropagation(data[:-labelSize])
            mse += self.getMeanSquareError(data[-labelSize:])
        return mse

    def execute(self, executionData):
        for data in executionData:
            out = self.forwardPropagation(data)[0]
            if out > 0:
                print("+1")
            else:
                print("-1")

    def trainValidateAndExecute(self, trainingsData, validationData, executionData, epochs, learningRate,
                                reducingLearningRate=0.0, goalMSE=0):
        for i in range(0, epochs):
            mse = self.validate(validationData, 1)
            if mse <= goalMSE:
                break
            self.train(trainingsData, 1, learningRate - (i * reducingLearningRate))

        self.execute(executionData)

    def animatedTrainValidateAndExecute(self, trainingsData, validationData, executionData, epochs, learningRate,
                                        reducingLearningRate=0.0, goalMSE=0):
        if len(trainingsData[0]) != 3:
            raise ValueError("trainings data must have a length of 3")
        if len(validationData[0]) != 3:
            raise ValueError("trainings data must have a length of 3")
        if len(executionData[0]) != 2:
            raise ValueError("execution data must have a length of 2")

        rows = 20
        columns = 20
        patternData = [[i / columns, j / rows] for i in range(-columns, columns) for j in range(-rows, rows)]

        plt.ion()
        fig, axis = plt.subplots(2)
        self.draw2DLabel(axis[1], trainingsData + validationData, 'v')

        for i in range(0, epochs):
            # Validation
            mse = self.validate(validationData, 1)
            plt.suptitle('Epoch: {}, mse: {}'.format(i, mse))
            plt.pause(0.0001)
            # draw Validation
            if i % 10 == 9:
                axis[0].clear()
                self.draw2DPattern(axis[0], patternData, 'o')
                self.draw2DLabel(axis[0], validationData, 'v', 1, "Orange", "Red")
                plt.pause(0.0001)

            if mse <= goalMSE:
                axis[0].clear()
                self.draw2DPattern(axis[0], patternData, 'o')
                self.draw2DLabel(axis[0], validationData, 'v', 1, "Orange", "Red")
                plt.suptitle('Epoch: {}, mse: {}, GOAL MSE REACHED!'.format(i, mse))
                plt.pause(3)
                break

            # Training
            self.train(trainingsData, 1, learningRate - (i * reducingLearningRate))

        plt.suptitle('Data executed, epoch{}'.format(epochs - 1))
        self.draw2DPattern(axis[1], executionData, 'o', 1, "SpringGreen", "Purple")
        plt.pause(3)
        self.execute(executionData)

    @staticmethod
    def draw2DLabel(axis, inputData, style='o', alpha=1, colorPos="Green", colorNeg="Indigo"):
        if len(inputData[0]) != 3:
            raise ValueError("input data must have a length of 3")

        # axis.clear()
        posX = []
        posY = []
        negX = []
        negY = []
        for data in inputData:
            if data[2] > 0:
                posX.append(data[0])
                posY.append(data[1])
            else:
                negX.append(data[0])
                negY.append(data[1])

        axis.plot(posX, posY, style, alpha=alpha, color=colorPos)

        axis.plot(negX, negY, style, alpha=alpha, color=colorNeg)

    def draw2DPattern(self, axis, inputData, style='o', alpha=1, colorPos="Green", colorNeg="Indigo"):
        if len(inputData[0]) != 2:
            raise ValueError("input data must have a length of 3")

        # axis.clear()

        posX = []
        posY = []
        negX = []
        negY = []
        for data in inputData:
            out = self.forwardPropagation(data[:2])[0]
            if out > 0:
                posX.append(data[0])
                posY.append(data[1])
            else:
                negX.append(data[0])
                negY.append(data[1])

        axis.plot(posX, posY, style, alpha=alpha, color=colorPos)

        axis.plot(negX, negY, style, alpha=alpha, color=colorNeg)


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
        self.__weights = {}
        if not self.__isInputLayer():
            # Add weights in size of anterior Layer plus Bias-Weight
            for n in self.__anteriorLayer:
                self.__weights[n] = Neuron.randomWeight()
            self.__weights['bias'] = Neuron.randomWeight()

    def setPosteriorLayer(self, posteriorLayer):
        self.__posteriorLayer = posteriorLayer

    def setOut(self, out):
        self.__out = out

    def setWeights(self, weights):
        self.__weights = weights

    def getNet(self):
        if not self.__isInputLayer():
            self.__net = 0
            for n in self.__anteriorLayer:
                self.__net += n.getOut() * self.getWeights(n)
            # add Bias Weight
            self.__net += self.__weights['bias']
        return self.__net

    def getOut(self):
        if not self.__isInputLayer():
            self.__out = self.__activationFunction(self.getNet())
        return self.__out

    def getWeights(self, weightKey=None):
        """
        get the weight or the whole weight dictionary of a specific neuron
        :param weightKey: Neuron or 'bias' that points to this associating weight
        :return: whole dictionary, if weightKey is none, or the associating weight
        """
        if weightKey is None:
            return self.__weights.copy()
        else:
            weight = self.__weights.get(weightKey)
            if weight is None:
                raise KeyError("key is not a member of weight dictionary")
            return weight

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
            for n in self.__posteriorLayer:
                posteriorDeltaSum += n.getDelta() * n.getWeights(self)
            self.__delta = self.__activationFunction(self.getNet(), True) * posteriorDeltaSum

        # update the weights
        for n in self.__weights.keys():
            if n == "bias":
                deltaWeight = learningRate * self.__delta
            else:
                deltaWeight = learningRate * self.__delta * n.getOut()
            self.__weights[n] += deltaWeight

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
        return random.randrange(-100, 100, 1) / 100  # 000


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
    epochs = 120
    learningRate = 0.1
    reducingLearningRate = 0.0001

    nn = NeuralNetwork()
    nn.addLayer(2, None)
    nn.addLayer(2, Neuron.tanh)
    nn.addLayer(1, Neuron.tanh)

    trainingData, validationData, executionData = loadInputData()

    nn.animatedTrainValidateAndExecute(trainingData, validationData, executionData, epochs, learningRate,
                               reducingLearningRate)


if __name__ == "__main__":
    main()
