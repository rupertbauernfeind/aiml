import math
import random
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self):
        self.__layers = []
        self.__inputLayer = None
        self.__outputLayer = None

    def add_layer(self, size, activation_function):
        """
        Adds a new Layer to your Neural Network.
        The first Layer is the InputLayer and the last one is the Output.
            - stores the __inputLayer and __outputLayer
            - stores the posterior Layer for each Layer
        :param size: Number of Neuron added to the Layer
        :param activation_function: static Neuron-Function. i.e. tanh()
        """
        anterior_layer = self.__outputLayer  # None if its the first input Layer
        new_layer = []

        # create Neurons and append to the Layer
        for i in range(0, size):
            n = Neuron(anterior_layer, activation_function)
            new_layer.append(n)

        self.__layers.append(new_layer)
        self.__outputLayer = new_layer
        self.__inputLayer = self.__layers[0]

        # set the actual Layer to the anterior layer as posterior layer
        if len(self.__layers) > 1:
            for n in anterior_layer:
                if isinstance(n, Neuron):
                    n.set_posterior_layer(new_layer)

    def forward_propagation(self, input_list):
        """
        Pass an Input through the Neural Network and get the final Output as a list
        :param input_list: Input-List, must have the size like the Input-Layer
        :return: Output Layer Value
        """
        if len(self.__layers) < 2:
            raise NameError("Not enough Layer to forwardPass. Minimum Input and Output Layer!")

        if len(input_list) != len(self.__inputLayer):
            raise NameError("Number of inputs doesn't match with input Layer")

        # set the Output of the Input Layer
        for i in range(0, len(self.__inputLayer)):
            n = self.__inputLayer[i]
            n.set_out(input_list[i])

        # pass forward through the hidden Layer to the Output Layer
        return [n.get_out() for n in self.__outputLayer]

    def get_mean_square_error(self, labels):
        mse = 0
        for i in range(0, len(self.__outputLayer)):
            n = self.__outputLayer[i]
            mse += 0.5 * ((labels[i] - n.get_out()) ** 2)
        return mse

    def backward_propagation(self, learning_rate, labels):
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
                    n.update_weight(learning_rate, labels[i])
            elif layer == self.__inputLayer:
                continue
            else:
                for n in layer:
                    n.update_weight(learning_rate)
        self.__layers.reverse()

    def train(self, trainings_data, label_size, learning_rate):
        for data in trainings_data:
            self.forward_propagation(data[:-label_size])
            self.backward_propagation(learning_rate, data[-label_size:])

    def validate(self, validation_data, label_size=1):
        mse = 0
        for data in validation_data:
            self.forward_propagation(data[:-label_size])
            mse += self.get_mean_square_error(data[-label_size:])
        return mse

    def execute(self, execution_data):
        for data in execution_data:
            out = self.forward_propagation(data)[0]
            if out > 0:
                print("+1")
            else:
                print("-1")

    def train_validate_and_execute(self, trainings_data, validation_data, execution_data, epochs, learning_rate,
                                   reducing_learning_rate=0.0, goal_mse=0):
        for i in range(0, epochs):
            mse = self.validate(validation_data, 1)
            if mse <= goal_mse:
                break
            self.train(trainings_data, 1, learning_rate - (i * reducing_learning_rate))

        self.execute(execution_data)

    def animated_train_validate_and_execute(self, trainings_data, validation_data, execution_data, epochs,
                                            learning_rate,
                                            reducing_learning_rate=0.0, goal_mse=0):
        if len(trainings_data[0]) != 3:
            raise ValueError("trainings data must have a length of 3")
        if len(validation_data[0]) != 3:
            raise ValueError("trainings data must have a length of 3")
        if len(execution_data[0]) != 2:
            raise ValueError("execution data must have a length of 2")

        rows = 20
        columns = 20
        pattern_data = [[i / columns, j / rows] for i in range(-columns, columns) for j in range(-rows, rows)]

        plt.ion()
        fig, axis = plt.subplots(2)
        self.draw2d_label(axis[1], trainings_data + validation_data, 'v')

        for i in range(0, epochs):
            # Validation
            mse = self.validate(validation_data, 1)
            plt.suptitle('Epoch: {}, mse: {}'.format(i, mse))
            plt.pause(0.0001)
            # draw Validation
            if i % 10 == 9:
                axis[0].clear()
                self.draw2d_pattern(axis[0], pattern_data, 'o')
                self.draw2d_label(axis[0], validation_data, 'v', 1, "Orange", "Red")
                plt.pause(0.0001)

            if mse <= goal_mse:
                axis[0].clear()
                self.draw2d_pattern(axis[0], pattern_data, 'o')
                self.draw2d_label(axis[0], validation_data, 'v', 1, "Orange", "Red")
                plt.suptitle('Epoch: {}, mse: {}, GOAL MSE REACHED!'.format(i, mse))
                plt.pause(3)
                break

            # Training
            self.train(trainings_data, 1, learning_rate - (i * reducing_learning_rate))

        plt.suptitle('Data executed, epoch{}'.format(epochs - 1))
        self.draw2d_pattern(axis[1], execution_data, 'o', 1, "SpringGreen", "Purple")
        plt.pause(3)
        self.execute(execution_data)

    @staticmethod
    def draw2d_label(axis, input_data, style='o', alpha=1, color_pos="Green", color_neg="Indigo"):
        if len(input_data[0]) != 3:
            raise ValueError("input data must have a length of 3")

        # axis.clear()
        pos_x = []
        pos_y = []
        neg_x = []
        neg_y = []
        for data in input_data:
            if data[2] > 0:
                pos_x.append(data[0])
                pos_y.append(data[1])
            else:
                neg_x.append(data[0])
                neg_y.append(data[1])

        axis.plot(pos_x, pos_y, style, alpha=alpha, color=color_pos)

        axis.plot(neg_x, neg_y, style, alpha=alpha, color=color_neg)

    def draw2d_pattern(self, axis, input_data, style='o', alpha=1, color_pos="Green", color_neg="Indigo"):
        if len(input_data[0]) != 2:
            raise ValueError("input data must have a length of 3")

        # axis.clear()

        pos_x = []
        pos_y = []
        neg_x = []
        neg_y = []
        for data in input_data:
            out = self.forward_propagation(data[:2])[0]
            if out > 0:
                pos_x.append(data[0])
                pos_y.append(data[1])
            else:
                neg_x.append(data[0])
                neg_y.append(data[1])

        axis.plot(pos_x, pos_y, style, alpha=alpha, color=color_pos)

        axis.plot(neg_x, neg_y, style, alpha=alpha, color=color_neg)


class Neuron:
    def __init__(self, anterior_layer, activation_function):
        """
        Create a new Neuron with the linked anterior Layer and the activation-function
        :param anterior_layer: Layer before the Neurons Layer
        :param activation_function: a function with one parameter, called for evaluating the Output
        """
        self.__out = None
        self.__net = None
        self.__delta = None
        self.__posteriorLayer = None
        self.__activationFunction = activation_function
        self.__anteriorLayer = anterior_layer
        self.__weights = {}
        if not self.__is_input_layer():
            # Add weights in size of anterior Layer plus Bias-Weight
            for n in self.__anteriorLayer:
                self.__weights[n] = Neuron.random_weight()
            self.__weights['bias'] = Neuron.random_weight()

    def set_posterior_layer(self, posterior_layer):
        self.__posteriorLayer = posterior_layer

    def set_out(self, out):
        self.__out = out

    def set_weights(self, weights):
        self.__weights = weights

    def get_net(self):
        if not self.__is_input_layer():
            self.__net = 0
            for n in self.__anteriorLayer:
                self.__net += n.get_out() * self.get_weights(n)
            # add Bias Weight
            self.__net += self.__weights['bias']
        return self.__net

    def get_out(self):
        if not self.__is_input_layer():
            self.__out = self.__activationFunction(self.get_net())
        return self.__out

    def get_weights(self, weight_key=None):
        """
        get the weight or the whole weight dictionary of a specific neuron
        :param weight_key: Neuron or 'bias' that points to this associating weight
        :return: whole dictionary, if weightKey is none, or the associating weight
        """
        if weight_key is None:
            return self.__weights.copy()
        else:
            weight = self.__weights.get(weight_key)
            if weight is None:
                raise KeyError("key is not a member of weight dictionary")
            return weight

    def get_delta(self):
        if self.__is_input_layer():
            raise NameError("InputLayer is not able to contain a delta Value")
        return self.__delta

    def update_weight(self, learning_rate=0.001, label=None):
        """
        Calculate the delta value for weight update: delta = f'(net) * (true - out)
        :param learning_rate:
        :param label: the should-Value if the Node is in the Output-Layer
        """
        if self.__is_input_layer():
            raise NameError("Weight update for InputLayer is not allowed")

        # compute the delta
        if self.__is_output_layer():
            if label is None:
                raise NameError("updateDelta needs a Should-Output-Value for training the Network")
            self.__delta = self.__activationFunction(self.get_net(), True) * (label - self.get_out())
        else:  # Hidden Layer
            posterior_delta_sum = 0
            for n in self.__posteriorLayer:
                posterior_delta_sum += n.get_delta() * n.get_weights(self)
            self.__delta = self.__activationFunction(self.get_net(), True) * posterior_delta_sum

        # update the weights
        for n in self.__weights.keys():
            if n == "bias":
                delta_weight = learning_rate * self.__delta
            else:
                delta_weight = learning_rate * self.__delta * n.get_out()
            self.__weights[n] += delta_weight

    def __is_input_layer(self):
        return self.__anteriorLayer is None or len(self.__anteriorLayer) == 0

    def __is_output_layer(self):
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
    def random_weight():
        return random.randrange(-100, 100, 1) / 100  # 000


def load_input_data():
    training_data = []
    execution_data = []

    input_strings = []

    while True:
        try:
            input_strings.append(input())
        except EOFError:
            break

    # sort the Data in training and execution Data
    for s in input_strings:
        value_strings = s.split(',')
        values = [float(value_strings[0]), float(value_strings[1])]
        if len(value_strings) > 2:
            values.append(float(value_strings[2]))
            if values[0] == 0 and values[1] == 0 and values[2] == 0:
                continue
            training_data.append(values)
        else:
            execution_data.append(values)

    # normalize the Data with max Magnitude per Input
    max_magnitude_x0 = 0
    max_magnitude_x1 = 0
    for data in training_data:
        if abs(float(data[0])) > max_magnitude_x0:
            max_magnitude_x0 = abs(float(data[0]))
        if abs(float(data[1])) > max_magnitude_x1:
            max_magnitude_x1 = abs(float(data[1]))

    for data in execution_data:
        if abs(float(data[0])) > max_magnitude_x0:
            max_magnitude_x0 = abs(float(data[0]))
        if abs(float(data[1])) > max_magnitude_x1:
            max_magnitude_x1 = abs(float(data[1]))

    for data in training_data:
        data[0] = data[0] / max_magnitude_x0
        data[1] = data[1] / max_magnitude_x1

    for data in execution_data:
        data[0] = data[0] / max_magnitude_x0
        data[1] = data[1] / max_magnitude_x1

    training_set = training_data[:int(len(training_data) * 0.7)]
    validation_set = training_data[int(len(training_data) * 0.7):]

    return training_set, validation_set, execution_data


def main():
    epochs = 400
    learning_rate = 0.1
    reducing_learning_rate = 0.0001

    nn = NeuralNetwork()
    nn.add_layer(2, None)
    nn.add_layer(6, Neuron.tanh)
    nn.add_layer(3, Neuron.tanh)
    nn.add_layer(1, Neuron.tanh)

    training_data, validation_data, execution_data = load_input_data()

    nn.animated_train_validate_and_execute(training_data, validation_data, execution_data, epochs, learning_rate,
                                           reducing_learning_rate)


if __name__ == "__main__":
    main()
