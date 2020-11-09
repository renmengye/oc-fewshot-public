# oc-fewshot-public

Code for paper *Wandering Within a World: Online Contextualized Few-Shot
Learning* [[arxiv](https://arxiv.org/abs/2007.04546)]

Authors: Mengye Ren, Michael L. Iuzzolino, Michael C. Mozer, Richard S. Zemel

## RoamingRooms Dataset

<span>
<img src="img/vid1.gif" width="250"><img src="img/vid2.gif" width="250"><img src="img/vid3.gif" width="250"></span>
<!--
![](img/vid1.gif | width=150)
![](img/vid2.gif | width=150)
![](img/vid3.gif | width=150) -->

Although our code base is MIT licensed, the RoamingRooms dataset is not since
it is derived from the Matterport3D dataset.

To download the RoamingRooms dataset, you first need to sign the agreement for
a non-commercial license
[here](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).

Then, you need to submit a request [here](https://forms.gle/rhha5EzUfh1SkvGM7).
We will manually approve your request afterwards.

For inquiries, please email:
[roaming-rooms@googlegroups.com](mailto:roaming-rooms@googlegroups.com)

The whole dataset is around 60 GB. It has 1.2M video frames with 7k unique
object instance classes. Please refer to our paper for more statistics of the
dataset.

If you have downloaded the full Matterport3D dataset already, check [here](sim)
for a script to generate RoamingRooms.

## System Requirements

Our code is tested on Ubuntu 18.04 with GPU capability. We provide docker
files for reproducible environments. We recommend at least 20GB CPU memory and
11GB GPU memory. 2-4 GPUs are required for multi-GPU experiments. Our code
is based on TensorFlow 2.

## Installation Using Docker (Recommended)

1. Install `protoc` from
   [here](http://google.github.io/proto-lens/installing-protoc.html).

2. Run `make` to build proto buffer configuration files.

3. Install `docker` and `nvidia-docker`.

4. Build the docker container using `./build_docker.sh`.

5. Modify the environment paths. You need to change `DATA_DIR` and `OURPUT_DIR`
   in `setup_environ.sh`. `DATA_DIR` is the main folder where datasets are
   placed and `OUTPUT_DIR` is the main folder where training models are saved.

## Installation Using Conda

1. Install `protoc` from
   [here](http://google.github.io/proto-lens/installing-protoc.html).

2. Run `make` to build proto buffer configuration files.

3. Modify the environment paths. You need to change `DATA_DIR` and `OURPUT_DIR`
   in `setup_environ.sh`. `DATA_DIR` is the main folder where datasets are
   placed and `OUTPUT_DIR` is the main folder where training models are saved.

4. Create a conda environment:
```
conda create -n oc-fewshot python=3.6
conda activate oc-fewshot
conda install pip
```

5. Install CUDA 10.1

6. Install OpenMPI 4.0.0

7. Install NCCL 2.6.4 for CUDA 10.1

8. Modify installation paths in `install.sh`

9. Run `install.sh`


## Setup Datasets

1. To set up the Omniglot dataset, run `script/download_omniglot.sh`. This
   script will download the Omniglot dataset to `DATA_DIR`.

2. To set up the Uppsala texture dataset (for spatiotemporal cue experiments),
   run `script/download_uppsala.sh`. This script will download the Uppsala
   texture dataset to `DATA_DIR`.

## RoamingOmniglot Experiments
To run training on your own, use the following command.

```
./run_docker.sh {GPU_ID} python -m fewshot.experiments.oc_fewshot \
  --config {MODEL_CONFIG_PROTOTXT} \
  --data {EPISODE_CONFIG_PROTOTXT} \
  --env configs/environ/roaming-omniglot-docker.prototxt \
  --tag {TAG} \
  [--eval]
```

* `MODEL_CONFIG_PROTOTXT` can be found in `configs/models`.
* `EPISODE_CONIFG_PROTOTXT` can be found in `configs/episodes`.
* `TAG` is the name of the saved checkpoint folder.
* When the model finishes training, add the `--eval` flag to evaluate.

For example, to train CPM on the semisupervised benchmark:

```
./run_docker.sh 0 python -m fewshot.experiments.oc_fewshot \
  --config configs/models/roaming-omniglot/cpm.prototxt \
  --data configs/episodes/roaming-omniglot/roaming-omniglot-150-ssl.prototxt \
  --env configs/environ/roaming-omniglot-docker.prototxt \
  --tag roaming-omniglot-ssl-cpm
```

All of our code is tested using GTX 1080 Ti with 11GB GPU memory. Note that the
above command uses a single GPU.  Our original experiments in the paper is
performed using two GPUs, with twice the batch size and doubled learning rate.
To run that setting, use the following command:

```
./run_docker_hvd_01.sh python -m fewshot.experiments.oc_fewshot_hvd \
  --config {MODEL_CONFIG_PROTOTXT} \
  --data {EPISODE_CONFIG_PROTOTXT} \
  --env configs/environ/roaming-omniglot-docker.prototxt \
  --tag {TAG}
```

## RoamingRooms Experiments
Below we include command to run experiments on RoamingRooms.
Our original experiments in the paper is performed using four GPUs, with batch
size to be 8. To run that setting, use the following command:

```
./run_docker_hvd_0123.sh python -m fewshot.experiments.oc_fewshot_hvd \
  --config {MODEL_CONFIG_PROTOTXT} \
  --data {EPISODE_CONFIG_PROTOTXT} \
  --env configs/environ/roaming-rooms-docker.prototxt \
  --tag {TAG}
```

When evaluate, use `--eval --usebest` to pick the checkpoint with the highest
validation performance.

## Results

Table 1: RoamingOmniglot Results (Supervised)

| Method             | AP      | 1-shot Acc.    | 3-shot Acc.    | Checkpoint        |
|:------------------:|:-------:|:--------------:|:--------------:|:-----------------:|
| OML-U              | 77.38   | 70.98 ± 0.21   | 89.13 ± 0.16   |[link](https://bit.ly/3lcAkPd)|
| OML-U++            | 86.85   | 88.43 ± 0.14   | 92.07 ± 0.14   |[link](https://bit.ly/32q2or3)|
| Online MatchingNet | 88.69   | 84.82 ± 0.15   | 95.55 ± 0.11   |[link](https://bit.ly/3edu8D4)|
| Online IMP         | 90.15   | 85.74 ± 0.15   | 96.66 ± 0.09   |[link](https://bit.ly/3ddMq5C)|
| Online ProtoNet    | 90.49   | 85.68 ± 0.15   | 96.95 ± 0.09   |[link](https://bit.ly/2YbeFxB)|
| CPM (Ours)         |**94.17**|**91.99** ± 0.11|**97.74** ± 0.08|[link](https://bit.ly/2IhQ2dx)|


Table 2: RoamingOmniglot Results (Semi-supervised)

| Method             | AP      | 1-shot  Acc.   | 3-shot Acc.    | Checkpoint        |
|:------------------:|:-------:|:--------------:|:--------------:|:-----------------:|
| OML-U              | 66.70   | 74.65 ± 0.19   | 90.81 ± 0.34   |[link](https://bit.ly/2UdezTf)|
| OML-U++            | 81.39   | 89.07 ± 0.19   | 89.40 ± 0.18   |[link](https://bit.ly/3mYo3OG)|
| Online MatchingNet | 84.39   | 88.77 ± 0.13   | 97.28 ± 0.17   |[link](https://bit.ly/3hDPDiA)|
| Online IMP         | 81.62   | 88.68 ± 0.13   | 97.09 ± 0.19   |[link](https://bit.ly/2NajIYO)|
| Online ProtoNet    | 84.61   | 88.71 ± 0.13   | 97.61 ± 0.17   |[link](https://bit.ly/2Yeex0x)|
| CPM (Ours)         |**90.42**|**93.18** ± 0.16|**97.89** ± 0.15|[link](https://bit.ly/3k8XOn2)|


Table 3: RoamingRooms Results (Supervised)

| Method             | AP      | 1-shot Acc.    | 3-shot Acc.    | Checkpoint        |
|:------------------:|:-------:|:--------------:|:--------------:|:-----------------:|
| OML-U              | 76.27   | 73.91 ± 0.37   | 83.99 ± 0.33   |[link](https://bit.ly/2GEa6WC)|
| OML-U++            | 88.03   | 88.32 ± 0.27   | 89.61 ± 0.29   |[link](https://bit.ly/3eIb14X)|
| Online MatchingNet | 85.91   | 82.82 ± 0.32   | 89.99 ± 0.26   |[link](https://bit.ly/3ddvAUG)|
| Online IMP         | 87.33   | 85.28 ± 0.31   | 90.83 ± 0.25   |[link](https://bit.ly/2YcdmP0)|
| Online ProtoNet    | 86.01   | 84.89 ± 0.31   | 89.58 ± 0.28   |[link](https://bit.ly/3fEh21J)|
| CPM (Ours)         |**89.14**|**88.39** ± 0.27|**91.31** ± 0.26|[link](https://bit.ly/3pbbwJC)|

Table 4: RoamingRooms Results (Semi-supervised)

| Method             | AP      | 1-shot Acc.    | 3-shot Acc.    | Checkpoint        |
|:------------------:|:-------:|:--------------:|:--------------:|:-----------------:|
| OML-U              | 63.40   | 70.67 ± 0.38   | 85.25 ± 0.56   |[link](https://bit.ly/32rnWU6)|
| OML-U++            | 81.90   | 84.79 ± 0.31   | 89.80 ± 0.47   |[link](https://bit.ly/38rCpDc)|
| Online MatchingNet | 78.99   | 80.08 ± 0.34   |**92.43** ± 0.41|[link](https://bit.ly/2YKD3VR)|
| Online IMP         | 75.36   | 84.57 ± 0.31   | 91.17 ± 0.43   |[link](https://bit.ly/37LI5FW)|
| Online ProtoNet    | 76.36   | 80.67 ± 0.34   | 88.83 ± 0.49   |[link](https://bit.ly/3hzXNbA)|
| CPM (Ours)         |**84.12**|**86.17** ± 0.30| 91.16 ± 0.44   |[link](https://bit.ly/3pclqus)|


## To-Do

* Add a data iterator based on PyTorch (contribution welcome).

## Citation

If you use our code, please consider cite the following:
* Mengye Ren, Michael L. Iuzzolino, Michael C. Mozer and Richard S. Zemel.
  Wandering Within a World: Online Contextualized Few-Shot Learning.
  *CoRR*, abs/2007.04546, 2020.

```
@article{ren20ocfewshot,
  author   = {Mengye Ren and
              Michael L. Iuzzolino and
              Michael C. Mozer and
              Richard S. Zemel},
  title    = {Wandering Within a World: Online Contextualized Few-Shot Learning},
  journal  = {CoRR},
  volume   = {abs/2007.04546},
  year     = {2020},
}
```
