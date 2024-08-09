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
import io
import time
MODEL_NAME = "ensemble_segmentation"


def process_output(img, pred, path_save, show=True):
    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
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
    if show:
        plt.show()
    fig.savefig(fname=path_save)
    plt.close()


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    """
    return np.fromfile(img_path, dtype="uint8")
def run(input_path, ip, port, path_save):
        # Load image.
    input_path = input_path
    # image_data = load_image(input_path)
    img_cv = cv2.imread(input_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv = cv2.resize(img_cv, INPUT_SIZE,  cv2.INTER_NEAREST)
    pil_img = Image.fromarray(img_cv, 'RGB')
    h = pil_img.size[0]
    w = pil_img.size[1]
    print("init size: ", pil_img.size)
    img = np.asarray(pil_img)
    print("image_data shape: ",img.shape)

    image_data = img.flatten()
    print("image_data shape: ",image_data.shape)

    image_data = np.expand_dims(image_data, axis=0)
    print("image_data shape: ",image_data.shape)
    start = time.time()
    num_test = 10
    print("testing ", num_test, ' ...')
    
    for i in range(num_test):
        # Setting up client
        url = f"{ip}:{port}"
        # print("Connecting to: ", url)
        client = grpcclient.InferenceServerClient(url=url)
        input_seg0 = grpcclient.InferInput("INPUT", image_data.shape, datatype="UINT8")

        input_seg0.set_data_from_numpy(image_data)
        mask = grpcclient.InferRequestedOutput("OUTPUT")
        # Call the server
        results = client.infer(model_name=MODEL_NAME, inputs=[input_seg0], outputs=[mask])
        mask = results.as_numpy('OUTPUT')[0].astype(np.uint8)

    print("Average time: ", (time.time() - start)/ num_test)
    # cv2.imshow("output", mask)
    # cv2.waitKey(3000)
    process_output(img, mask, path_save)
    # return pred_seg.shape

if __name__ == "__main__":
    input_path = 'data/1.jpg'
    ip="localhost"
    port = 1997
    path_save='./outputs/seg.jpg'
    run(input_path, ip, port, path_save)
