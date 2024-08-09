import argparse
import sys

import gevent.ssl
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from utils.color_classifier import ColorClassifier
from torchvision import transforms
import cv2
from PIL import Image
import time
from utils.preprocess import *

MODEL_NAME = "clothes_segmentation"
NUM_CHANNEL = 3
INPUT_SIZE = (256, 256)

def test_infer(
    input_data,
    http_client,
    headers=None,
    request_compression_algorithm=None,
    response_compression_algorithm=None,):
    
    
    input = httpclient.InferInput("intput", input_data.shape, "FP32")
    # Initialize the data
    input.set_data_from_numpy(input_data, binary_data=False)
    output = httpclient.InferRequestedOutput("output", binary_data=True)
    results = http_client.infer(
        model_name= MODEL_NAME,
        inputs=[input],
        outputs=[output],
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm,
    )
    return results.as_numpy('output')


if __name__ == "__main__":
    color_file = "./clients/ColorDef.xml"
    color_classifier = ColorClassifier(color_file)

    input_path = "./images/1.jpg"
    data_transform = get_transform()
    input_numpy, img_cv= preprocess(input_path, data_transform)
    # input_numpy = np.zeros((1, 3, INPUT_SIZE[0], INPUT_SIZE[1]), dtype=np.float32)

    server_url = 'localhost:8000'
    http_client = httpclient.InferenceServerClient(url=server_url)

    num_iter = 20
    start = time.time()
    for i in range(num_iter):
        output = test_infer(input_numpy, http_client)    
    print("Average time: ", (time.time() - start)/num_iter)
    image = img_cv
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    upper_colors, upper_mask = color_classifier.get_upper_color(hsv_img, image, output)
    lower_colors, lower_mask = color_classifier.get_lower_color(hsv_img, image, output)

    print(output.shape)
    print(upper_colors)
    print(lower_colors)
