FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates git wget vim cmake ninja-build protobuf-compiler \
		libprotobuf-dev build-essential cmake curl ca-certificates screen \
        freeglut3-dev xvfb\
  	&& rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda install -y python=3.8 \
    && /opt/conda/bin/conda install -y numpy jupyter matplotlib pandas tqdm \
    && /opt/conda/bin/conda install -y pytorch==1.8.1 cudatoolkit=10.2 -c pytorch -c nvidia \
    && /opt/conda/bin/conda install -y -c conda-forge ezc3d multiprocess tensorboard tensorboardx easydict \
    && /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN python -m pip install wandb

RUN git clone https://github.com/beyaldiz/fairmotion.git \
    && cd fairmotion \
    && python -m pip install -e .

#Port for Jupyter, Tensorboard
EXPOSE 8888
EXPOSE 6006

