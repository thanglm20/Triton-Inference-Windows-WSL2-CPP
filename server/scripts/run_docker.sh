 #!/bin/bash

docker run --gpus=1 -it -p8000:8000 -p8001:8001 -p8002:8002 \
    -v $(pwd)/model_repository:/models naiz_triton_server:2.0 \
    tritonserver --model-repository=/models
