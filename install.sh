# Script to install with conda.
#
# Author: Mengye Ren (mren@cs.toronto.edu)
#
# Modify the path below:
# ------------------------------------------------
CUDA=/pkgs/cuda-10.1
OPEN_MPI=/pkgs/openmpi-4.0.0
NCCL=/pkgs/nccl_2.6.4-1+cuda10.1_x86_64
# ------------------------------------------------

export PATH=$PATH:$OPEN_MPI/bin
pip install numpy \
        tensorflow-gpu==2.2.0rc1 \
        keras \
        h5py
HOROVOD_GPU_ALLREDUCE=NCCL \
        HOROVOD_GPU_BROADCAST=NCCL \
        HOROVOD_WITH_TENSORFLOW=1 \
        HOROVOD_CUDA_HOME=$CUDA \
        HOROVOD_CUDA_INCLUDE=$CUDA/include \
        HOROVOD_CUDA_LIB=$CUDA/lib64 \
        HOROVOD_NCCL_HOME=$NCCL \
        HOROVOD_NCCL_INCLUDE=$NCCL/include HOROVOD_NCCL_LIB=$NCCL/lib \
        pip install --no-cache-dir horovod==0.19.2
pip install tqdm \
        opencv-python==3.4.2.17 \
        pandas \
        tensorflow_addons==0.8.3 \
        scikit-learn matplotlib

source setup_environ.sh
ln -s $OUTPUT_DIR ./results
ln -s $DATA_DIR ./data
