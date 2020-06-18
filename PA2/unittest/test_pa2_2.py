import unittest
import sys
import os
from pathlib import Path
from PA2.src.utils import get_project_root


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
        root_path = get_project_root(True)
        app_path = root_path + str(Path("PA2/src/pa2_2.py"))
        input_path = root_path + str(Path("PA2/unittest/example/input{}.txt".format(n)))
        output_path = root_path + str(Path("PA2/unittest/_output{}.txt".format(n)))
        valid_path = root_path + str(Path("PA2/unittest/example/output{}.txt".format(n)))

        if sys.platform == 'linux':
            os.system("bash " + app_path + " < " + input_path + " > " + output_path)
        else:  # Windows
            os.system(app_path + " < " + input_path + " > " + output_path)

        o1 = "o1"
        o2 = "o2"
        try:

            output_file1 = open(valid_path, "r")
            output_file2 = open(output_path, "r")

            o1 = output_file1.read()
            o2 = output_file2.read()

            output_file1.close()
            output_file2.close()
        finally:
            if os.path.isfile(output_path):
                os.remove(output_path)

        self.assertEqual(o1.strip(), o2.strip())


if __name__ == '__main__':
    unittest.main()
