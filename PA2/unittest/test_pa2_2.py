import unittest
import os


class TestPa2v2(unittest.TestCase):
    def test_example1(self):
        outputN = "_output1.txt"
        os.system("C:\\Users\\ruper\\PycharmProjects\\AIML\\PA2\\src\\pa2_2.py" + " < example\\input1.txt > " + outputN)

        try:
            outputFile1 = open("example\\output1.txt", "r")
            outputFile2 = open(outputN, "r")

            o1 = outputFile1.read()
            o2 = outputFile2.read()

            outputFile1.close()
            outputFile2.close()
        finally:
            if os.path.isfile(outputN):
                os.remove(outputN)

        s = []
        v = o1.split("\n")
        for i in v:
            s.append(int(i))
        print(s)
        self.assertEqual(o1.strip(), o2.strip())

    def test_example2(self):
        outputN = "_output2.txt"
        os.system("C:\\Users\\ruper\\PycharmProjects\\AIML\\PA2\\src\\pa2_2.py" + " < example\\input2.txt > " + outputN)

        try:
            outputFile1 = open("example\\output2.txt", "r")
            outputFile2 = open(outputN, "r")

            o1 = outputFile1.read()
            o2 = outputFile2.read()

            outputFile1.close()
            outputFile2.close()

        finally:
            if os.path.isfile(outputN):
                os.remove(outputN)

        self.assertEqual(o1.strip(), o2.strip())

    def test_example3(self):
        outputN = "_output3.txt"
        os.system("C:\\Users\\ruper\\PycharmProjects\\AIML\\PA2\\src\\pa2_2.py" + " < example\\input3.txt > " + outputN)

        try:
            outputFile1 = open("example\\output3.txt", "r")
            outputFile2 = open(outputN, "r")

            o1 = outputFile1.read()
            o2 = outputFile2.read()

            outputFile1.close()
            outputFile2.close()
        finally:
            if os.path.isfile(outputN):
                os.remove(outputN)

        self.assertEqual(o1.strip(), o2.strip())


if __name__ == '__main__':
    unittest.main()
