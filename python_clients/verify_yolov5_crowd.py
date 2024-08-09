
import argparse
from numpy import random
import cv2
import torch
import numpy as np
from utils.yolov5_postprocess import  non_max_suppression, scale_coords, xyxy2xywh
import tritonclient.grpc as grpcclient
import time

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tl=1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label and False:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class TritonClient:
    def __init__(self, url):
        self.url = url
        self.in_shape = [1, 737280]
        self.client = grpcclient.InferenceServerClient(url=self.url)

        self.input_data = grpcclient.InferInput("INPUT", self.in_shape, datatype="UINT8")
        self.model_name = "ensemble_yolov5_crowd"

    def request_triton(self, img):
        self.input_data.set_data_from_numpy(img)
        # Call the server
        output_data = grpcclient.InferRequestedOutput("OUTPUT")
        results = self.client.infer(model_name=self.model_name, inputs=[self.input_data], outputs=[output_data])
        output_data = results.as_numpy('OUTPUT')

        return output_data

# def request_triton(url, img):
#     MODEL_NAME = "ensemble_yolov5_crowd"
#     client = grpcclient.InferenceServerClient(url=url)
#     input_data = grpcclient.InferInput("INPUT", img.shape, datatype="UINT8")
#     input_data.set_data_from_numpy(img)
#     output_data = grpcclient.InferRequestedOutput("OUTPUT")
#     # Call the server
#     results = client.infer(model_name=MODEL_NAME, inputs=[input_data], outputs=[output_data])
#     output_data = results.as_numpy('OUTPUT')
#     return output_data

INPUT_SHAPE = (640, 384)
def main(opt):
    client = TritonClient(opt.url)
    input_video_path = 'data/1.mp4'
    output_video_path = 'data/1_result.mp4'
    webcam = False
    view_img = opt.view_img

    # Create a VideoCapture object to read the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Define the codec for the output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the width and height of the input video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to write the output video file
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # Check if the input video file was opened successfully
    if not cap.isOpened():
        print("Error opening input video file")

    # Get names and colors
    names = ['person', 'head']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    while cap.isOpened():
        # Read the next frame from the input video file
        ret, im0s = cap.read()
        
        if not ret :
            break
        # Padded resize
        img = cv2.resize(im0s, INPUT_SHAPE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = img.flatten()
        input_data =  np.expand_dims(input_data, axis=0)

        start = time.time()
        # pred = executor.infer(img)[3]
        # pred = request_triton(opt.url, input_data)
        pred = client.request_triton(input_data)
        pred = torch.from_numpy(pred)
        print(pred.shape)
        print("Time request: " , time.time() - start)
        

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', im0s
            img_shape = [384, 640]
            print("det shape: ", det.shape )
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_shape, det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):              
                    if view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # if 'head' == names[int(cls)] :
                            # text = f"Person : {n}"
                            # cv2.putText(im0, text, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                        if opt.heads or opt.person:
                            if 'head' in label and opt.heads:
                                plot_one_box(xyxy, im0, label=label, color=(0, 255, 0), line_thickness=4)
                            if 'person' in label and opt.person:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        else:
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Write the modified frame to the output video file using the write() method of the VideoWriter object
        out.write(im0s)

        # Display the modified frame
        cv2.imshow('Video Playback', im0s)

        # Wait for 25 milliseconds and check for a key press
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        


    print("Done")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
    parser.add_argument('--url', type=str, default='172.31.123.244:8001', help='model.pt path(s)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', default=True, help='display results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--person', action='store_true', default=True, help='displays only person')
    parser.add_argument('--heads', action='store_true',  default=True, help='displays only person')
    opt = parser.parse_args()
    print(opt)
    main(opt)