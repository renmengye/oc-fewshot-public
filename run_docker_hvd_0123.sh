source setup_environ.sh
ARGS="${@:1}"

if [ -z "$ARGS" ]; then
  echo "Must provide program command"
  exit 1
fi

docker run -it --rm \
    --mount type=bind,source=$DATA_DIR,target=/root/mount/$NAME/data \
    --mount type=bind,source=$OUTPUT_DIR,target=/root/mount/$NAME/results \
    --volume $CODE_FOLDER:/root/mount/$NAME \
    -w /root/mount/$NAME \
    --gpus '"device=0,1,2,3"' \
    $DOCKERNAME horovodrun -np 4 -H localhost:4 $ARGS
