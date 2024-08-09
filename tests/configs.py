
import os
import configparser

class Configs:
    _instance = None

    @staticmethod
    def get_instance(config_file="config.ini"):
        if Configs._instance is None:
            config = Configs(config_file=config_file)
            Configs._instance = config
        return Configs._instance
    
    def __init__(self,  config_file):
        config = self.load_config(config_file)
        self.image_test = config.get("image_test")
        self.results_file=config.get("results_file")
        self.ip =config.get("ip")
        self.grpc_port = config.get("grpc_port")
        self.http_port = config.get("http_port")
        self.num_thread = config.getint("num_thread")
        self.num_loop=config.getint("num_loop")
        self.output_shape=eval(config.get("output_shape"))
        self.output_seg=config.get("output_seg")
        self.output_clothes=config.get("output_clothes")

    def load_config(self, config_file):
        print("Loeading config file: ", config_file)
        if not os.path.isfile(config_file): 
            print("Not found config file, exit !!!")
            os._exit(1)
        config = configparser.ConfigParser()
        config.read(config_file)
        return config["COMMON"]



