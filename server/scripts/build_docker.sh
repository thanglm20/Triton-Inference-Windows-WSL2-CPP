
#!/bin/bash
echo "Start building docker image from NVIDIA ..."
sudo rm -fr /usr/local/lib/docker
docker build -f Dockerfile -t naiz_triton_server:1.0 .
echo "Done"