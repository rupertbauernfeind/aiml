class Status:
    undiscovered = 1
    discovered = 2
    visited = 3


class Vertex:
    def __init__(self, name):
        self.name = name
        self.cost = 99999
        self.status = Status.undiscovered
        self.neighborKeys = {}
        self.parent = ''

    def addNeighbor(self, toName: str, cost: int):
        if toName not in self.neighborKeys:
            self.neighborKeys[toName] = cost
            return True
        else:
            return False


class Graph:
    # Save all vertices in a dictionary
    vertices = {}

    def addVertex(self, fromName, toName, cost):
        if fromName not in self.vertices:
            vertex = Vertex(fromName)
            self.vertices[fromName] = vertex
        if toName not in self.vertices:
            vertex = Vertex(toName)
            self.vertices[toName] = vertex
        v = self.vertices[fromName]
        if isinstance(v, Vertex):
            v.addNeighbor(toName, cost)
            return True
        else:
            return False

    def printGraph(self):
        for key in sorted(list(self.vertices.keys())):
            v = self.vertices[key]
            if isinstance(v, Vertex):
                print(v.name + " connected with: " + str(v.neighborKeys) + "\n" +
                      "Status: " + str(v.status) + " Parent: " + v.parent + " cost: " + str(v.cost))

    def dijkstraSearch(self, sKey):
        Q = list()
        s = self.vertices[sKey]
        if isinstance(s, Vertex):
            s.cost = 0
            s.parent = ""
            self.dis(Q, s)
        while len(Q) > 0:
            fromVertex = Q.pop(0)
            self.dis(Q, fromVertex)

    def dis(self, Q: list, fromVertex: Vertex):
        for toName in fromVertex.neighborKeys:
            cost = fromVertex.neighborKeys[toName]
            toVertex = self.vertices[toName]
            if isinstance(toVertex, Vertex):
                # if toVertex.status == Status.undiscovered:
                #     toVertex.status = Status.discovered

                if toVertex.cost > fromVertex.cost + cost:
                    toVertex.cost = fromVertex.cost + cost
                    toVertex.parent = fromVertex.name
                    Q.append(toVertex)

    def printDijkstra(self):
        for key in sorted(self.vertices.keys()):
            v = self.vertices[key]
            if isinstance(v, Vertex):
                c = str(v.cost)
                resultTxt = v.name
                while v.parent != "":
                    v = self.vertices[v.parent]
                    resultTxt = resultTxt + "-" + v.name
                resultTxt = resultTxt + "-" + c
                print(resultTxt)


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
            graph.addVertex(vertexInfo[0], vertexInfo[2], int(vertexInfo[4:]))

        except EOFError:
            break

    graph.dijkstraSearch(firstVertexInfo)
    graph.printDijkstra()


if __name__ == "__main__":
    main()
