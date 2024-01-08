## How to use the docker container

Build the image using

`docker build -t learning .`

Make sure to install the `nvidia-container-toolkit` package.
To spin up and attach to a container from the image, run

`sudo docker run --gpus all -it --rm --name learning-container -v (pwd):/home/learning/src learning`

To attach to an already running container, use

`docker exec -it learning-container bash`

You might want to use this to start the tensorboard server so that you can follow the learning logs and statistics.

