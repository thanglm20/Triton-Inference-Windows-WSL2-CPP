


FROM nvcr.io/nvidia/tritonserver:23.12-py3

RUN python3 -m pip install --upgrade pip
RUN pip install torch==2.0.1 torchvision==0.15.2
RUN pip install opencv-python
RUN python3 -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install libgl1

WORKDIR /opt/tritonserver