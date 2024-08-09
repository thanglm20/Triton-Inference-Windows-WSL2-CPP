import cv2
import json
import numpy as np
import time
from numpy.testing import assert_almost_equal
from PIL import Image
import tritonclient.grpc as grpcclient
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from utils.preprocess import *

MODEL_NAME = "clothes_segmentation"
NUM_CHANNEL = 3
INPUT_SIZE = 256


def process_output(img, pred, path_save):
    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    classes = np.array(('Background',  # always index 0
                        'Hat', 'Hair', 'Glove', 'Sunglasses',
                        'UpperClothes', 'Dress', 'Coat', 'Socks',
                        'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                        'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                        'Right-leg', 'Left-shoe', 'Right-shoe',))
    colormap = [(0, 0, 0),
                (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
                (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ]
    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    h, w, _ = pred.shape
    pred = pred.reshape((h, w))
    # show image
    ax0.set_title('img')
    ax0.imshow(img)
    ax1.set_title('pred')
    mappable = ax1.imshow(pred, cmap=cmap, norm=norm)
    # plt.show()
    fig.savefig(fname=path_save)
    plt.close()


def run(input_path, ip, port, path_save):
        # Load image.
    # input_img = cv2.imread('images/1.jpg')
    input_path = input_path
    data_transform = get_transform()
    input_numpy, img_cv= preprocess(input_path, data_transform)
    # Setting up client
    url = f"{ip}:{port}"
    # print("Connecting to: ", url)
    client = grpcclient.InferenceServerClient(url=url)
    input_seg = grpcclient.InferInput("intput", input_numpy.shape, datatype="FP32")
    input_seg.set_data_from_numpy(input_numpy.astype("float32"))
    pred = grpcclient.InferRequestedOutput("output")

    # Call the server
    results = client.infer(model_name=MODEL_NAME, inputs=[input_seg], outputs=[pred])
    pred_seg = results.as_numpy('output')
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    process_output(img_cv, pred_seg, path_save)


    upper_mask_ids = [5,6,7]
    lower_mask_ids = [9, 10, 12]
    upper_mask = np.isin(pred_seg, upper_mask_ids)
    upper_mask = np.where(upper_mask>0, 255, 0).astype(np.uint8)
    lower_mask = np.isin(pred_seg, lower_mask_ids)
    lower_mask = np.where(lower_mask>0, 255, 0).astype(np.uint8)
    mask = lower_mask | upper_mask
    print("shape: ", upper_mask.shape)
    cv2.imshow("mask", mask)
    cv2.waitKey(3000)
    return pred_seg.shape

if __name__ == "__main__":
    input_path = 'data/1.jpg'
    ip="localhost"
    port = 1997
    path_save='./outputs/seg.jpg'
    run(input_path, ip, port, path_save)
