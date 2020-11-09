source setup_environ.sh
ARGS="${@:1}"

if [ -z "$ARGS" ]; then
  echo "Must provide program command"
  exit 1
fi

### We assume that the GPUs are partitioned in an order so no deadlock.
GPU1=-1
GPU2=-1
while [ $GPU1 == -1 ]
do
    GPU1=$(/usr/bin/python gpu_lock.py --ids 0)
    if [ $GPU1 == -1 ]; then
      sleep 5;
    else
        while [ $GPU2 == -1 ]
        do
            GPU2=$(/usr/bin/python gpu_lock.py --ids 1)
            if [ $GPU2 == -1  ]; then
                sleep 5;
            fi
        done
    fi
done

docker run -it --rm \
    --mount type=bind,source=$DATA_DIR,target=/root/mount/$NAME/data \
    --mount type=bind,source=$OUTPUT_DIR,target=/root/mount/$NAME/results \
    --volume $CODE_FOLDER:/root/mount/$NAME \
    -w /root/mount/$NAME \
    --gpus '"device=0,1"' \
    $DOCKERNAME horovodrun -np 2 -H localhost:2 $ARGS
