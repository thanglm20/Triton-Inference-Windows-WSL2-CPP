# Client programs testing Triton Inference Server 
This project includes: 
- Triton Server Inference deployed on Windows WSL 2 for some models: Yolov5, Segmentation, Color Extraction, ...
- Triton Client using gRPC, C++ in VS2019

## Start Triton Server
```

$ docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v {model_repository_path}:/models nvcr.io/nvidia/tritonserver:23.11-py3 tritonserver --model-repository=/models

example: docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/thanglm/TritonServer/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:23.11-py3 tritonserver --model-repository=/models
```
##
```
ThangLM Triton server: 
ip=172.31.123.244
grpc_port=8001
http_port=8000

```
## Install requirements
```
$ pip install -r requirements.txt
```
## Run Python Clients
```
1. Verify Triton Server
    $ python .\clients\verify_read.py

2. Verify color clothes
    $ python .\clients\verify_segmention.py

3.  Verify color clothes
    $ python .\clients\verify_color.py --ip={} --time_display={}
    Default: python .\clients\verify_color.py --ip=localhost --time_display=3

4. Stress test many request concurrently
    $ python .\clients\stress_test_requests.py --ip=localhost --num_thread={} --num_requests={}
    Default: python .\clients\stress_test_requests.py --num_thread=10 --num_requests=100
```


### Run Unitests
```
Change configs in tests/configs.ini
and then run all testcases
$ python -m unittest discover .\tests\ -p '*_test.py' -v
```

## Run CPP Clients
### Requirements

```
1. VS2019
2. GRPC prebuilt by VS2019
- Download here: https://github.com/thommyho/Cpp-gRPC-Windows-PreBuilts
- Export GRPC_ROOT_DIR to environment variables
3. Opencv
```

### Run
.\ColorExtractor.exe --image_path=[] --colordef=[] --url=[] --num_test=200

Ex: .\ColorExtractor.exe --image_path=./input/3.jpg --colordef=ColorDef.xml --url=172.31.123.244:8001 --num_test=200
