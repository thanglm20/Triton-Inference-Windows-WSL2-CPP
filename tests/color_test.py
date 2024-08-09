import unittest
from tests.configs import Configs
from .. clients.verify_color import *


class TestClothesColor(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)
        self.config = Configs.get_instance("./tests/configs.ini")

    def read_json_rs(self):
        with open(self.config.results_file, 'r') as f:
            data = json.load(f)
        upper_colors = data['upper']
        lower_colors = data['lower']
        return upper_colors, lower_colors
    
    def test_clothes_color(self):
        output, upper_colors,lower_colors = run(self.config.image_test, self.config.ip, self.config.grpc_port, self.config.num_loop, self.config.output_clothes)
        real_upper_colors,real_lower_colors = self.read_json_rs()
        self.assertEqual(json.dumps(upper_colors), json.dumps(real_upper_colors))
        self.assertEqual(json.dumps(lower_colors), json.dumps(real_lower_colors))


if __name__ == '__main__':
    unittest.main()