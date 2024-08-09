import unittest
from tests.configs import Configs
from clients.verify_ready import *


class TestReady(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)
        self.config = Configs.get_instance("./tests/configs.ini")

    def test_ready(self):
        ready = is_ready(self.config.ip, self.config.http_port)
        self.assertTrue(ready)


if __name__ == '__main__':
    unittest.main()