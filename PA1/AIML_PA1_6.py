class Status:
    undiscovered = 1
    discovered = 2
    visited = 3


class Vertex:
    def __init__(self, name):
        self.name = name
        self.cost = 99999
        self.status = Status.undiscovered
        self.neighborKeys = list()
        self.parent = ''

    def addNeighbors(self, neighborString):
        for c in neighborString:
            if c not in self.neighborKeys:
                self.neighborKeys.append(c)


class Graph:
    # Save all vertices in a dictionary
    vertices = {}

    def addVertex(self, name, neighborString):
        if name not in self.vertices:
            vertex = Vertex(name)
            vertex.addNeighbors(neighborString)
            self.vertices[name] = vertex
            return True
        else:
            return False

    def printGraph(self):
        for key in sorted(list(self.vertices.keys())):
            v = self.vertices[key]
            if isinstance(v, Vertex):
                print(v.name + " connected with: " + str(v.neighborKeys) + "\n" +
                      "Status: " + str(v.status) + " Parent: " + v.parent + " cost: " + str(v.cost))

    def breadthFirstSearch(self, sKey):
        Q = list()
        s = self.vertices[sKey]
        s.color = Status.discovered
        s.cost = 0
        s.parent = ""
        txt = self.bfs(Q, s)
        while len(Q) > 0:
            v = Q.pop(0)
            txt = txt + self.bfs(Q, v)
        print(txt)

    def bfs(self, Q: list, s: Vertex):
        s.status = Status.visited
        for key in s.neighborKeys:
            n = self.vertices[key]
            if isinstance(n, Vertex):
                if n.status == Status.undiscovered:
                    Q.append(n)
                    n.status = Status.discovered
                    if n.cost > s.cost + 1:
                        n.cost = s.cost + 1
                        n.parent = s.name
        return s.name


def main():
    graph = Graph()
    firstVertexInfo = ""
    while True:
        try:
            vertexInfo = input()
            if firstVertexInfo == "":
                firstVertexInfo = vertexInfo[0]

            if len(vertexInfo) <= 1:
                break
            graph.addVertex(vertexInfo[0], vertexInfo[2:])

        except EOFError:
            break

    graph.breadthFirstSearch(firstVertexInfo)


if __name__ == "__main__":
    main()
