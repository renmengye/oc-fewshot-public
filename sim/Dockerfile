# Base image
FROM nvidia/cudagl:9.2-devel-ubuntu18.04

# Install cudnn
ENV CUDNN_VERSION 7.6.4.38
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip \
    libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda9.2 \
    && \
    apt-mark hold libcudnn7 &&\
    rm -rf /var/lib/apt/lists/*

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    doxygen \
    curl \
    libjsoncpp-dev \
    libepoxy-dev \
    libglm-dev \
    libosmesa6 \
    libosmesa6-dev \
    libglew-dev \
    libopencv-dev \
    python-opencv
    # python3-setuptools \
    # python3-dev \
    # python3-pip

# Install conda
RUN curl -o ~/miniconda.sh -LO  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install \
        pyyaml \
        mkl \
        mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.7
RUN /bin/bash -c ". activate habitat; pip install --upgrade pip"
RUN /bin/bash -c ". activate habitat; conda install numpy scipy ipython pandas jupyterlab matplotlib"

# Setup habtiat-sim
RUN git clone https://github.com/facebookresearch/habitat-sim.git
# RUN /bin/bash -c ". activate habitat; cd habitat-sim; git checkout 62bc01be8cbc148eb099e96fa0b123e16b21f95f; python setup.py install --headless"
RUN /bin/bash -c ". activate habitat; cd habitat-sim; git checkout d281a086d7cc24a4269276c21d6c55e3aa6c989f; python setup.py install --headless --with-cuda"

# Install challenge specific habitat-api
RUN git clone https://github.com/facebookresearch/habitat-api.git
RUN /bin/bash -c ". activate habitat; cd habitat-api; git checkout 0985c6ffd17557150488d238d79574c60612faa9; pip install -e ."

# Install MatterSim
RUN /bin/bash -c ". activate habitat; pip install opencv-python==3.4.2.17 \
    opencv-contrib-python==3.4.2.17 \
    networkx==2.2"
RUN git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git
RUN /bin/bash -c ". activate habitat; \
    cd Matterport3DSimulator; \
    git checkout 5331d70d8ce2ddfba2a1c0ec53b0e1db28182bc5; \
    mkdir build; \
    cd build; \
    cmake -DEGL_RENDERING=ON ..; \
    make"
ENV PYTHONPATH="/Matterport3DSimulator/build:"

# Other python libraries
RUN /bin/bash -c ". activate habitat; conda install h5py seaborn"
RUN /bin/bash -c ". activate habitat; pip install torchvision pyrr"
RUN /bin/bash -c ". activate habitat; conda install numba"
RUN apt-get install -y tmux
RUN /bin/bash -c "echo $'unbind C-b\n \
set-option -g prefix C-a\n \
bind-key C-a send-prefix\n \
'>/root/.tmux.conf"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
RUN /bin/bash -c ". activate habitat; pip install tensorflow-gpu==2.2.0rc1 tensorflow_addons==0.8.3"

# CMD ["/root/mount/fewshot-lifelong/sim/start_jupyter.sh"]
