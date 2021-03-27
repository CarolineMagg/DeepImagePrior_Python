# DeepImagePrior in Python
Deep Image Prior (DIP) implementation in python with tensorflow/keras. <br>

This repository provides a Python (Tensorflow/Keras) implementation based on [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior) by Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor [paper](https://arxiv.org/pdf/1711.10925.pdf).

## Docker & Requirements
You can use [Docker](https://www.docker.com/) to setup your environment. For installation guide see [Install Docker](https://docs.docker.com/get-docker/). <br> 

The docker image contains:
* Python 3.6
* Tensorflow 2.4.1 
* Jupyter notebooks (minimal requirement)
* Pycharm

### How to build the docker image:
1. clone the github repository 
2. go to Dockerfile: ``` cd dockerfiles/ ```
3. download [Pycharm](https://www.jetbrains.com/de-de/pycharm/download/#section=linux) and copy deb file into "dockerfiles/installer.tgz" 
4. change Dockerfile to use your user name instead of *caroline* 
5. Build docker image: ``` docker build --tag python-tf-docker:1.00 .``` 

### How to run a docker container:
(Note: Change *user* to your home folder name.)
* Run jupyter docker container: <br>
```docker run -it --gpus all --name jupyter_notebook --rm -v /home/user/:/tf/workdir -p 8888:8888 python-tf-docker:1.00``` <br>

Recommended: modify sh files to fit your settings
