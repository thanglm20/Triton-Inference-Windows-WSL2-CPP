
# Triton Inference Client

# Requirements

```
1. VS2019
2. GRPC prebuilt by VS2019
- Download here: https://github.com/thommyho/Cpp-gRPC-Windows-PreBuilts
- Export GRPC_ROOT_DIR to environment variables
3. Opencv




# Run
.\ColorExtractor.exe --image_path=[] --colordef=[] --url=[] --num_test=200

Ex: .\ColorExtractor.exe --image_path=./input/3.jpg --colordef=ColorDef.xml --url=172.31.123.244:8001 --num_test=200
```