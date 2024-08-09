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
from utils.color_classifier import ColorClassifier
from utils.preprocess import *

MODEL_NAME = "clothes_segmentation"
NUM_CHANNEL = 3
INPUT_SIZE = (256, 256)
PORT = 1997


def show_image(img, pred):
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
    fig.savefig(fname="output_seg.jpg")
    plt.close()

def draw_results( img, outputs_table, type='upper'):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # fontScale 
        fontScale = 0.5
        # Blue color in BGR 
        color = (0, 0, 0) 
        # Line thickness of 2 px 
        thickness = 1
        if type == 'upper':
            pos = [5, 15]
            img = cv2.putText(img, "Upper color prediction", (pos[0], pos[1]), font,  
                            fontScale, color, thickness, cv2.LINE_AA)
            pos[1] += 16
        elif type== 'lower':
            pos = [5, img.shape[0]//2 + 15]
            img = cv2.putText(img, "Lower color prediction", (pos[0], pos[1]), font,  
                            fontScale, color, thickness, cv2.LINE_AA)
            pos[1] += 16

        img = cv2.line(img,(0,img.shape[0]//2),(100,img.shape[0]//2),(255,0,0),1)
        for cl in outputs_table:
            color_name = cl + ": " + str(outputs_table[cl]) + "%"
            img = cv2.putText(img, color_name, (pos[0], pos[1]), font,  
                            fontScale, color, thickness, cv2.LINE_AA)
            pos[1] += 16
        return img 

def draw_outputs(img, upper_colors, upper_mask, lower_colors, lower_mask, debug_seg = True):
        img = cv2.resize(img,INPUT_SIZE)
        output = cv2.copyMakeBorder(img, 0, 0, 200, 0,  cv2.BORDER_CONSTANT, None, (255, 255, 255))
        #get upper color
        upper_seg = cv2.bitwise_and(img, img, mask=upper_mask)
        output = draw_results(output, upper_colors, 'upper')
        lower_seg = cv2.bitwise_and(img, img, mask=lower_mask)
        output = draw_results(output, lower_colors, 'lower')
        if debug_seg:
            output = cv2.hconcat([output, upper_seg])
            output = cv2.hconcat([output, lower_seg])
        return output

def run(img_path, ip, port, num_loop, save_path):
    color_file = "./clients/ColorDef.xml"
    color_classifier = ColorClassifier(color_file)
    input_path = img_path
    data_transform = get_transform()
    input_numpy, img_cv= preprocess(input_path, data_transform)
    # Setting up client
    server_url = f'{ip}:{port}'
    client = grpcclient.InferenceServerClient(url=server_url)
    input_seg = grpcclient.InferInput("intput", input_numpy.shape, datatype="FP32")
    input_seg.set_data_from_numpy(input_numpy.astype("float32"))
    pred = grpcclient.InferRequestedOutput("output")

    # ignore first time
    results = client.infer(model_name=MODEL_NAME, inputs=[input_seg], outputs=[pred])
    # request to server
    NUMBER_ITER = num_loop
    start  = time.time()
    for i in range(NUMBER_ITER):
        results = client.infer(model_name=MODEL_NAME, inputs=[input_seg], outputs=[pred])
    print("Average time: ", (time.time() - start)/NUMBER_ITER)
    pred_seg = results.as_numpy('output')
    # image = cv2.resize(img_cv, INPUT_SIZE, cv2.INTER_NEAREST)
    image = img_cv
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    upper_colors, upper_mask = color_classifier.get_upper_color(hsv_img, image, pred_seg)
    lower_colors, lower_mask = color_classifier.get_lower_color(hsv_img, image, pred_seg)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output = draw_outputs(image, upper_colors, upper_mask,lower_colors, lower_mask)
    cv2.imwrite(save_path, output)
    return output, upper_colors, lower_colors

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
    parser.add_argument('--image_path', type=str, default='images/1.jpg')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=1997)
    parser.add_argument('--time_display', type=int, default=3)
    parser.add_argument('--num_loop', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./outputs/clothes_color.jpg')

    args = parser.parse_args()
   
    output, upper_colors,lower_colors  = run(args.image_path, args.ip, args.port, args.num_loop, args.save_path)
    print("upper: ", upper_colors)
    print("lower: ", lower_colors)

    print(f"Displaying in {args.time_display}s ...")
    cv2.imshow("output", output)    
    cv2.waitKey(args.time_display*1000) # 3s
    print("Done")

