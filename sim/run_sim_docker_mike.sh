export MATTERPORT_DATA_DIR=/hdd/mliuzzolino/Matterport/v1
export FOLDER=/home/michael/fewshot-lifelong
export HDD_DIR=/hdd/mliuzzolino/fewshot-lifelong

DEFAULT_PORT=8990
if [ -z "$1" ]
  then
    PORT=$DEFAULT_PORT
else
    PORT=$1
fi

echo "Launching on port $PORT"

docker run --runtime=nvidia -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/fewshot-lifelong/sim/data/v1 \
    --mount type=bind,source=$HDD_DIR,target=/root/mount/fewshot-lifelong/sim/data/fewshot \
    --volume $FOLDER:/root/mount/fewshot-lifelong \
    --env QT_X11_NO_MITSHM=1 \
    -p $PORT:$PORT \
    fewshot-lifelong
