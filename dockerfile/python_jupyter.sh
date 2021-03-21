docker run -it --gpus all --name jupyter_notebook --rm \
        	-v /home/caroline/:/tf/workdir \
	       	-p 8888:8888\
	       	python-tf-docker:1.00
