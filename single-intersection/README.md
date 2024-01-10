## How to use the docker container

Build the image using

`docker build -t learning .`

In order to use you GPU from inside the docker container, make sure to install
the `nvidia-container-toolkit` package. Please see the [installation
guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for more information. To spin up and attach to a container from the image, run

`docker run --gpus all -it --rm --name learning-container -v $(pwd):/home/learning/src learning`

You still have to install the necessary packages (still need to automate this in the build) via Poetry by running

`cd src && poetry install`

The Python environment created by Poetry can be activated by running

`poetry shell`

After this, you are able to run any scripts, for example

`cd single-intersection && python dqn.py`

To attach to an already running container, use

`docker exec -it learning-container bash`

You might want to use this to start the tensorboard server with 

`tensorboard --logdir runs --bind_all`

so that you can follow the learning logs and statistics. To find the IP address
of the running container use

`docker inspect learning-container | grep IPAddress`

