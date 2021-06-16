# DeepImagePrior in Python
Deep Image Prior (DIP) implementation in python with tensorflow/keras. <br>

This repository provides a Python (Tensorflow/Keras) implementation based on [Deep Image Prior](https://arxiv.org/pdf/1711.10925.pdf) by Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor. The implementation is based on the [Pytorch version](https://dmitryulyanov.github.io/deep_image_prior) provided with the original paper.

## Docker & Requirements
You can use [Docker](https://www.docker.com/) to setup your environment. For installation guide see [Install Docker](https://docs.docker.com/get-docker/). <br> 

The docker image contains:
* Python 3.6
* Tensorflow 2.4.1 
* Jupyter notebooks (minimal requirement)

### How to build the docker image:
1. clone the github repository 
2. go to Dockerfile: ``` cd dockerfiles/ ```
3. change Dockerfile to use your user name instead of *caroline* 
4. Build docker image: ``` docker build --tag python-tf-docker:1.00 .``` 

### How to run a docker container:
(Note: Change *user* to your home folder name.)
* Run jupyter docker container: <br>
``` docker run -it --gpus all --name jupyter_notebook --rm -v /home/user/:/tf/workdir -p 8888:8888 python-tf-docker:1.00 ``` <br>

Recommended: modify sh files to fit your settings

## Data

The natural images are taken from the implementation provided alongside the [original paper](https://github.com/DmitryUlyanov/deep-image-prior).

The microscopy data is taken from [ISBI 2012 Challenge Dataset](http://brainiac2.mit.edu/isbi_challenge/home) [1,2].

[1] Albert Cardona, Stephan Saalfeld, Stephan Preibisch, Benjamin Schmid, Anchi Cheng, Jim Pulokas, Pavel Tomancak and Volker Hartenstein (10, 2010), "An Integrated Micro- and Macroarchitectural Analysis of the Drosophila Brain by Computer-Assisted Serial Section Electron Microscopy", PLoS Biol (Public Library of Science) 8 (10): e1000502, doi:10.1371/journal.pbio.1000502 <br>
[2] Ignacio Arganda-Carreras, Srinivas C. Turaga, Daniel R. Berger, Dan Ciresan, Alessandro Giusti, Luca M. Gambardella, Jürgen Schmidhuber, Dmtry Laptev, Sarversh Dwivedi, Joachim M. Buhmann, Ting Liu, Mojtaba Seyedhosseini, Tolga Tasdizen, Lee Kamentsky, Radim Burget, Vaclav Uher, Xiao Tan, Chanming Sun, Tuan D. Pham, Eran Bas, Mustafa G. Uzunbas, Albert Cardona, Johannes Schindelin, and H. Sebastian Seung. Crowdsourcing the creation of image segmentation algorithms for connectomics. Frontiers in Neuroanatomy, vol. 9, no. 142, 2015.
