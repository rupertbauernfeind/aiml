class Status:
    undiscovered = 1
    discovered = 2


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
    cost = 0
    resultTxt = ""

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

    def depthFirstSearch(self, sKey):
        s = self.vertices[sKey]
        s.cost = self.cost
        s.parent = ""
        s.status = Status.discovered
        self.resultTxt = self.resultTxt + s.name
        self.dfs(s)
        print(self.resultTxt)

    def dfs(self, s: Vertex):
        for key in s.neighborKeys:
            n = self.vertices[key]
            if isinstance(n, Vertex):
                if n.status != Status.undiscovered:
                    continue
                n.status = Status.discovered
                self.cost = self.cost + 1
                n.cost = self.cost
                self.resultTxt = self.resultTxt + n.name
                n.parent = s.name
                self.dfs(n)


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

    graph.depthFirstSearch(firstVertexInfo)


if __name__ == "__main__":
    main()
