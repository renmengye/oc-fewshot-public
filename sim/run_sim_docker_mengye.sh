export MATTERPORT_DATA_DIR=/mnt/research/datasets/matterport3d/v1
export OUTPUT_DIR=/mnt/research/output/fewshot-lifelong/sim
export HDD_DIR=/mnt/research/datasets/matterport3d/fewshot
FOLDER=$PWD/..

# docker run -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
# docker run --runtime=nvidia -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#     --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/fewshot-lifelong/sim/data/v1 \
#     --mount type=bind,source=$OUTPUT_DIR,target=/root/mount/fewshot-lifelong/sim/output \
#     --mount type=bind,source=$HDD_DIR,target=/root/mount/fewshot-lifelong/sim/data/fewshot \
#     --volume $FOLDER:/root/mount/fewshot-lifelong \
#     --env QT_X11_NO_MITSHM=1 \
#     -p 8990:8990 \
#     fewshot-lifelong
    # renmengye/fewshot-lifelong:simulator
docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/fewshot-lifelong/sim/data/v1 \
    --mount type=bind,source=$OUTPUT_DIR,target=/root/mount/fewshot-lifelong/sim/output \
    --mount type=bind,source=$HDD_DIR,target=/root/mount/fewshot-lifelong/sim/data/fewshot \
    --volume $FOLDER:/root/mount/fewshot-lifelong \
    --env QT_X11_NO_MITSHM=1 \
    -p 8990:8990 \
    --gpus "device=0" \
    -w /root/mount/fewshot-lifelong/sim \
    fewshot-lifelong
