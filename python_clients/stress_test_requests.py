from typing import Any
import cv2
import json
import numpy as np
import time
from numpy.testing import assert_almost_equal
from PIL import Image
import tritonclient.grpc as grpcclient
from threading import Thread
import threading
from torchvision import transforms
from utils.color_classifier import ColorClassifier

MODEL_NAME = "clothes_segmentation"
NUM_CHANNEL = 3
INPUT_SIZE = 256
PORT = 1997

class TestThread(Thread):
    def __init__(self, ip, num_requests):
        Thread.__init__(self)
        self.ip = ip
        self.num_request = num_requests
        self.path_image = 'images/1.jpg'
        color_file = "./clients/ColorDef.xml"
        self.color_classifier = ColorClassifier(color_file) 
        self.fps = 30

    def get_transform(self):
        transform_image_list = [
        # transforms.Resize((256, 256), 3),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(transform_image_list)
    
    def preprocess(self, img_path, data_transform):
        ori_img = Image.open(img_path)
        img = data_transform(ori_img)
        img_cv = cv2.imread(img_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_cv = cv2.resize(img_cv, (INPUT_SIZE, INPUT_SIZE),  cv2.INTER_NEAREST)
        return img.unsqueeze(dim=0).cpu().numpy(), img_cv

    def run(self):
        thread_id = threading.get_native_id()
        # Setting up client
        client = grpcclient.InferenceServerClient(url=f"{self.ip}:{PORT}")
        for i in range(self.num_request):
            data_transform = self.get_transform()
            input_numpy, img_cv = self.preprocess(self.path_image, data_transform)
            # input_numpy = np.zeros(input_numpy.shape, dtype=np.float32)
            start = time.time()
            print(input_numpy.shape)
            input_seg = grpcclient.InferInput("intput", input_numpy.shape, datatype="FP32")
            input_seg.set_data_from_numpy(input_numpy.astype("float32"))
            pred = grpcclient.InferRequestedOutput("output")
            results = client.infer(model_name=MODEL_NAME, inputs=[input_seg], outputs=[pred])
            request_time = time.time() -start
            pred_seg = results.as_numpy('output')
            print(f"Thread id {thread_id} is testing: [seq:{i},\t request time = {request_time}s,\t total time = {time.time()-start}s] ")
            hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
            upper_colors, upper_mask = self.color_classifier.get_upper_color(hsv_img, img_cv, pred_seg)
            lower_colors, lower_mask = self.color_classifier.get_lower_color(hsv_img, img_cv, pred_seg)
            print(f"\t upper: ", upper_colors)
            print(f"\t lower: ", lower_colors)
            time.sleep(1/self.fps) # ~30fps
            # print("Output shape: ", pred_seg.shape)
        print(f"Thread id {thread_id} testing done!!!")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--num_thread', type=int, default=10)
    parser.add_argument('--num_requests', type=int, default=100)

    args = parser.parse_args()
    ip = args.ip
    num_thread = args.num_thread
    num_requests = args.num_requests

    testers = []
    for i in range(num_thread):
        t = TestThread(ip, num_requests)
        t.daemon = True
        testers.append(t)
    for tester in testers:
        tester.start()


    while True:
        try:
            for tester in testers:
                tester.join()
                testers.remove(tester)
            if(len(testers) == 0):
                
                break
            time.sleep(0.2)
        except KeyboardInterrupt:
            print("Interrupted!!!")
            break
    print("Done all!")

