N=10000                          # Total number of episodes
S=100                            # Total number of sequence
nframe=200                       # Total number of frames
m=5                              # Total number of processes
output='h5_data'                 # Output directory
nps=$(( $N / $S ))               # Total number of episode per sequence
n=$(( $nps / $m ))               # Number of episodes per processes
m1=$(( $m - 1 ))
S1=$(( $S - 1 ))
pid_arr=()
echo "Number of episodes: $N"
echo "Number of sequences: $S"
echo "Number of episodes per sequence: $nps"
echo "Number of processes: $m"
echo "Number of episodes per process: $n"

for s in $(seq 0 1 $S1)
do
  for i in $(seq 0 1 $m1)
  do
    seed=$(( $s * 43691 + $i * 17 ))
    start=$(( $i * $n + $s * $nps ))
    echo $i $seed $start
    ./generate_episodes_mengye.sh $seed $start $n $nframe $output &
    pid=$!
    pid_arr[$i]=$pid
    echo "Launched $pid"
  done
  for item in ${pid_arr[*]}
  do
    while kill -0 $item ; do
       echo "Process $item is still active..."
       sleep 2
    done
  done
  echo "Sequence $s all done"
done
echo "Fin."
