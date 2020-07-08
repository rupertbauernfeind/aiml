import random
import math
import statistics


class VectorQuantization:
    def __init__(self, clusters):
        self.clusters = []
        for i in range(0, clusters):
            self.clusters.append(Cluster(-0.75 + i * (2 / clusters), 0))

    def center_weights(self, _patterns):
        pX_list = [pattern[0] for pattern in _patterns]
        pY_list = [pattern[1] for pattern in _patterns]
        center1X = statistics.mean(pX_list)
        center1Y = statistics.mean(pY_list)
        self.clusters[0].w1 = center1X
        self.clusters[0].w2 = center1Y

        for i in range(1, len(self.clusters)):
            most_dist = 0
            most_dist_pattern = _patterns[0]
            for pat in _patterns:
                dist = 0
                for j in range(0, i):
                    c = self.clusters[j]
                    dist += c.get_distance(pat[0], pat[1])
                if dist >= most_dist:
                    most_dist = dist
                    most_dist_pattern = pat
            self.clusters[i].w1 = most_dist_pattern[0]
            self.clusters[i].w2 = most_dist_pattern[1]

    def train(self, x, y):
        # Get the winning Cluster
        c_win = self.__get_winning_cluster(x, y)
        c_win.update_weight(x, y)
        return c_win

    def __get_winning_cluster(self, x, y):
        c_win = None
        c_win_distance = 0
        for c in self.clusters:
            c_distance = c.get_distance(x, y)
            if c_win is None or c_distance <= c_win_distance:
                c_win = c
                c_win_distance = c_win.get_distance(x, y)
        return c_win

    def get_cluster_center_sum(self):
        center_sum = 0
        for c in self.clusters:
            center_sum += c.w1 + c.w2
        return center_sum

    @staticmethod
    def random_of_range(min_range, max_range):
        return random.randrange(min_range * 100, max_range * 100, 1) / 100  # 000


class Cluster:
    def __init__(self, w1, w2):
        self.w1 = w1
        self.w2 = w2

    def get_distance(self, x, y):
        dx = self.w1 - x
        dy = self.w2 - y
        return math.sqrt((dx ** 2) + (dy ** 2))

    def update_weight(self, x, y):
        self.w1 += (x - self.w1) * 0.001
        self.w2 += (y - self.w2) * 0.001


def load_normalized_input_data():
    """
    Get patterns with the size of 20x10
    :return: A list with training patterns and a evaluation pattern
    """
    input_patterns = []

    while True:
        try:
            input_string = input()
            if input_string.strip() != "":
                input_patterns.append(input_string)
        except EOFError:
            break

    numberOfCluster = input_patterns[0]

    # filter pattern
    _patterns = []
    for i in range(1, len(input_patterns)):
        _p = input_patterns[i].split(',')
        t = (float(_p[0]), float(_p[1]))
        _patterns.append(t)

    # normalize pattern
    max_abs = 0
    for _p in _patterns:
        if abs(_p[0] > max_abs):
            max_abs = abs(_p[0])
        if abs(_p[1] > max_abs):
            max_abs = abs(_p[1])

    # for i in range(0, len(_patterns)):
    #    x_norm = _patterns[i][0] / max_abs
    #    y_norm = _patterns[i][1] / max_abs
    #    _patterns[i] = (x_norm, y_norm)

    return int(numberOfCluster), _patterns, max_abs


if __name__ == "__main__":

    numberOfClusters, patterns, max_val = load_normalized_input_data()
    vectorQuantization = VectorQuantization(numberOfClusters)
    vectorQuantization.center_weights(patterns)
    for _ in range(0, 300):
        for p in patterns:
            c_win = vectorQuantization.train(p[0], p[1])

    print(vectorQuantization.get_cluster_center_sum())

