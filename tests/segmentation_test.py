import unittest
from tests.configs import Configs
from clients.verify_segmentation import *


class TestSegmentation(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)
        self.config = Configs.get_instance("./tests/configs.ini")
    def test_output_shape(self):
        output_shape = run(self.config.image_test, self.config.ip, self.config.grpc_port, self.config.output_seg)
        self.assertEqual(self.config.output_shape, output_shape)

    
if __name__ == '__main__':
    unittest.main()