class HopFieldNetwork:

    def __init__(self, size_rows, size_columns):
        # Generate all Neurons and weights
        self.__neurons = [[Neuron() for _ in range(0, size_columns)] for _ in range(0, size_rows)]
        self.__size_rows = size_rows
        self.__size_columns = size_columns

    # region INTERFACE
    def learn_pattern(self, pattern):
        self.__set_values_with_pattern(pattern)
        self.__update_weight()

    def find_pattern_async(self, pattern):
        self.__set_values_with_pattern(pattern)
        self.__update_pattern()
        return self.__get_pattern_from_values()

    def find_pattern_sync(self, pattern):
        ...

    def __update_pattern(self):
        for i in range(0, self.__size_rows):
            for j in range(0, self.__size_columns):
                self.__neurons[i][j].update_value(self.__neurons, self.__size_rows, self.__size_columns)

    def __get_pattern_from_values(self):
        pattern = [[0 for _ in range(0, self.__size_columns)] for _ in range(0, self.__size_rows)]
        for i in range(0, self.__size_rows):
            for j in range(0, self.__size_columns):
                pattern[i][j] = self.__neurons[i][j].value
        return pattern

    def __update_weight(self):
        for i in range(0, self.__size_rows):
            for j in range(0, self.__size_columns):
                self.__neurons[i][j].update_weight(self.__neurons, self.__size_rows, self.__size_columns)

    def __set_values_with_pattern(self, pattern):
        for i in range(0, self.__size_rows):
            for j in range(0, self.__size_columns):
                self.__neurons[i][j].value = pattern[i][j]
    # endregion


class Neuron:

    def __init__(self):
        self.value = 0
        self.__weights = None

    def update_weight(self, neurons, rows, columns):
        for i in range(0, rows):
            for j in range(0, columns):
                n = neurons[i][j]
                if self.__weights is None:
                    self.__weights = [[0 for _ in range(0, columns)] for _ in range(0, rows)]

                if n == self:
                    continue

                self.__weights[i][j] += self.value * n.value

    def update_value(self, neurons, rows, columns):
        for i in range(0, rows):
            for j in range(0, columns):
                n = neurons[i][j]
                self.value += n.value * self.__weights[i][j]
        self.value /= abs(self.value)

    @staticmethod
    def identity(x, derivative=False):
        if derivative:
            return 1
        else:
            return x


def string_to_pattern(pattern_string):
    if isinstance(pattern_string, str):
        pattern = []
        row = []
        for c in pattern_string:
            if c == '.':
                row.append(-1)
            elif c == '*':
                row.append(1)
            elif c == '\n':
                pattern.append(row)
                row = []
            else:
                raise ValueError("Unknown char for pattern")
        return pattern
    raise ValueError("Expected pattern as string")


def pattern_to_string(pattern):
    if isinstance(pattern, list):
        pattern_string = ""
        for row in pattern:
            for i in row:
                if i == -1:
                    pattern_string += "."
                elif i == 1:
                    pattern_string += "*"
                else:
                    raise ValueError("Unknown integer in pattern")
            pattern_string += '\n'
        return pattern_string
    raise ValueError("Expected pattern as Array")


def load_input_data():
    """
    Get patterns with the size of 20x10
    :return: A list with training patterns and a evaluation pattern
    """
    train_patterns = []

    # read all inputs
    input_string = ""
    while True:
        try:
            input_string += input() + '\n'
        except EOFError:
            break

    # sort the Data in training and execution Data
    if isinstance(input_string, str):
        train_patterns = input_string.split("---\n")
    if len(train_patterns) != 2:
        raise Exception("There should be exact one eval_pattern")

    eval_patterns = train_patterns[1].split('-\n')
    train_patterns = train_patterns[0].split('-\n')
    return train_patterns, eval_patterns


if __name__ == "__main__":
    # Create HopFieldNetwork
    hf_network = HopFieldNetwork(10, 20)

    # Read input-data
    train_p, eval_p = load_input_data()

    # learn data in hf_network
    for p in train_p:
        hf_network.learn_pattern(string_to_pattern(p))

    # find data in hf_network
    for i in range(0, len(eval_p)):
        p = eval_p[i]
        for _ in range(0, 10):
            p = hf_network.find_pattern_async(string_to_pattern(p))
            p = pattern_to_string(p)
        if i < len(eval_p) - 1:
            print(p + '-')
        else:
            print(p[:-1])

