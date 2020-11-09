DATA_DIR=/mnt/research/datasets/matterport3d/fewshot/h5_data_new2
TARGET_DIR=/mnt/research/datasets/matterport3d/fewshot/h5_data_new2_tar
mkdir -p $TARGET_DIR
PWD2=$PWD
cd $DATA_DIR

for f in *;
do
  echo 'f' $f
  # filename=$(basename -- "$f")
  # echo 'filename' $filename

  tar -cf $f.tar $f
  mv $f.tar $TARGET_DIR
  # extension="${filename##*.}"
  # filename="${filename%.*}";
  # echo $filename;
  # unzip -o $f -d $filename
done;
cd $PWD2
