FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libpng16-16 \
	libzmq3-dev \
	pkg-config \
	libsm6 \
	libxext6 \
	graphviz \
	python3.6-tk \
	git \
	vim \
	apt-utils \
	libxtst-dev libxext-dev libxrender-dev libfreetype6-dev libfontconfig1 libgtk2.0-0 libxslt1.1 libxxf86vm1 \
	wget \
	curl \
	software-properties-common \
	make \
	gcc \
	g++ \
	bzip2 \
	hdf5-tools \
	unzip \
	gfortran \
	curl \
	sudo \
	libx11-xcb1 libasound2 x11-apps libice6 libsm6 libxaw7 libxft2 libxmu6 libxpm4 libxt6 x11-apps xbitmaps
	
# install python packages	
RUN pip3.6 install pandas \	
	matplotlib \
	scikit-image \
	opencv-python==4.1.0.25 \
	pydot 

RUN adduser --disabled-password --gecos '' caroline
RUN usermod -aG sudo caroline
USER caroline
WORKDIR /home/caroline/

RUN jupyter notebook --generate-config
COPY jupyter_notebook_config.py /home/caroline/.jupyter/

WORKDIR /home/caroline/
