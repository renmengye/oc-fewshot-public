source setup_environ.sh

GPUFLAG=$1
if [ "$GPUFLAG" == "auto" ]; then
  # Automatically pick GPU.
  GPU=-1
  while [ $GPU == -1 ]
  do
    GPU=$(/usr/bin/python gpu_lock.py --id)
    if [ $GPU == -1 ]; then
      sleep 5;
    fi
  done
else
  # Manually specify GPU.
  GPU=-1
  while [ $GPU == -1 ]
  do
  GPU=$(/usr/bin/python gpu_lock.py --ids $GPUFLAG)
    if [ $GPU == -1 ]; then
      sleep 5;
    fi
  done
fi
ARGS="${@:2}"

if [ -z "$GPU" ]; then
  echo "Must provide GPU ID"
  exit 1
fi

if [ -z "$ARGS" ]; then
  echo "Must provide program command"
  exit 1
fi

docker run -it --rm \
    --mount type=bind,source=$DATA_DIR,target=/root/mount/$NAME/data \
    --mount type=bind,source=$OUTPUT_DIR,target=/root/mount/$NAME/results \
    --volume $CODE_FOLDER:/root/mount/$NAME \
    -w /root/mount/$NAME \
    --gpus "device=$GPU" \
    $DOCKERNAME $ARGS
