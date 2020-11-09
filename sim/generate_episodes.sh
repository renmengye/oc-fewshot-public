source setup_environ.sh
FOLDER=$PWD/..
SEED=$1
START=$2
N=$3
NFRAME=$4
OUTPUTDIR=$5
PROG="python generate_dataset.py \
--seed $SEED \
--start $START \
--n_episodes $N \
--resize_width 160 \
--resize_height 120 \
--n_frames_generate 600 \
--n_frames_save $NFRAME \
--output_dir $OUTPUTDIR"

docker run --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/oc-fewshot/sim/data/v1 \
    --mount type=bind,source=$OUTPUT_DIR,target=/root/mount/oc-fewshot/sim/output \
    --mount type=bind,source=$FEWSHOT_DATA_DIR,target=/root/mount/oc-fewshot/sim/data/fewshot \
    --volume $FOLDER:/root/mount/oc-fewshot \
    --env QT_X11_NO_MITSHM=1 \
    --gpus "device=0" \
    -w /root/mount/oc-fewshot/sim \
    oc-fewshot-sim \
    /bin/bash -c ". activate habitat; $PROG"
