import unittest
import os
from PA2.src import pa2_2


class TestPa2v2(unittest.TestCase):

    def test_example1(self):
        self.exampleN(1)

    def test_example2(self):
        self.exampleN(2)

    def test_example3(self):
        self.exampleN(3)

    def test_example4(self):
        self.exampleN(4)

    def test_example5(self):
        self.exampleN(5)

    def test_example6(self):
        self.exampleN(6)

    def exampleN(self, n):

        os.system(os.path.realpath(pa2_2.__file__) + " < example\\input{}.txt > _output{}".format(n, n))

        o1 = "o1"
        o2 = "o2"
        try:
            outputFile1 = open("example\\output{}.txt".format(n), "r")
            outputFile2 = open("_output{}".format(n), "r")

            o1 = outputFile1.read()
            o2 = outputFile2.read()

            outputFile1.close()
            outputFile2.close()
        finally:
            if os.path.isfile("_output{}".format(n)):
                os.remove("_output{}".format(n))

        self.assertEqual(o1.strip(), o2.strip())


if __name__ == '__main__':
    unittest.main()
