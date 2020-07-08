import unittest
import unittest
import sys
import os
from pathlib import Path
from PA2.src.utils import get_project_root


class MyTestCase(unittest.TestCase):

    def test_example1_1(self):
        self.exampleN(2, 1)

    def exampleN(self, part, sub_part):

        root_path = get_project_root(True)
        app_path = root_path + str(Path("PA3/src/pa3_{}.py".format(part)))
        input_path = root_path + str(Path("PA3/unittest/example/input{}_{}.txt".format(part, sub_part)))
        output_path = root_path + str(Path("PA3/unittest/_output{}_{}.txt".format(part, sub_part)))
        valid_path = root_path + str(Path("PA3/unittest/example/output{}_{}.txt".format(part, sub_part)))

        if sys.platform == 'linux':
            os.system("python3 " + app_path + " < " + input_path + " > " + output_path)
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
            ...

        self.assertEqual(o1.strip(), o2.strip())


if __name__ == '__main__':
    unittest.main()
