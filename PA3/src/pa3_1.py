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


# region BASIC_MATH
def matrix_addition(m_a, m_b):
    """
    Adds two matrices and returns the sum
        :param m_a: The first matrix
        :param m_b: The second matrix

        :return: Matrix sum
    """
    # Section 1: Ensure dimensions are valid for matrix addition
    rows_a = len(m_a)
    cols_a = len(m_a[0])
    rows_b = len(m_b)
    cols_b = len(m_b[0])
    if rows_a != rows_b or cols_a != cols_b:
        raise ArithmeticError('Matrices are NOT the same size.')

    # Section 2: Create a new matrix for the matrix sum
    C = zeros_matrix(rows_a, cols_b)

    # Section 3: Perform element by element sum
    for i in range(rows_a):
        for j in range(cols_b):
            C[i][j] = m_a[i][j] + m_b[i][j]

    return C


def matrix_subtraction(m_a, m_b):
    """
    Subtracts matrix B from matrix A and returns difference
        :param m_a: The first matrix
        :param m_b: The second matrix

        :return: Matrix difference
    """
    # Section 1: Ensure dimensions are valid for matrix subtraction
    rows_a = len(m_a)
    cols_a = len(m_a[0])
    rows_b = len(m_b)
    cols_b = len(m_b[0])
    if rows_a != rows_b or cols_a != cols_b:
        raise ArithmeticError('Matrices are NOT the same size.')

    # Section 2: Create a new matrix for the matrix difference
    C = zeros_matrix(rows_a, cols_b)

    # Section 3: Perform element by element subtraction
    for i in range(rows_a):
        for j in range(cols_b):
            C[i][j] = m_a[i][j] - m_b[i][j]

    return C


def matrix_multiply(m_a, m_b):
    """
    Returns the product of the matrix A * B
        :param m_a: The first matrix - ORDER MATTERS!
        :param m_b: The second matrix

        :return: The product of the two matrices
    """
    # Section 1: Ensure A & B dimensions are correct for multiplication
    rows_a = len(m_a)
    cols_a = len(m_a[0])
    rows_b = len(m_b)
    cols_b = len(m_b[0])
    if cols_a != rows_b:
        raise ArithmeticError(
            'Number of A columns must equal number of B rows.')

    # Section 2: Store matrix multiplication in a new matrix
    m_c = zeros_matrix(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0
            for ii in range(cols_a):
                total += m_a[i][ii] * m_b[ii][j]
            m_c[i][j] = total

    return m_c


def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have

        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def transpose(m_m):
    """
    Returns a transpose of a matrix.
        :param m_m: The matrix to be transposed

        :return: The transpose of the given matrix
    """
    # Section 1: if a 1D array, convert to a 2D array = matrix
    if not isinstance(m_m[0], list):
        m_m = [m_m]

    # Section 2: Get dimensions
    rows = len(m_m)
    cols = len(m_m[0])

    # Section 3: m_t is zeros matrix with transposed dimensions
    m_t = zeros_matrix(cols, rows)

    # Section 4: Copy values from M to it's transpose m_t
    for i in range(rows):
        for j in range(cols):
            m_t[j][i] = m_m[i][j]

    return m_t
# endregion
