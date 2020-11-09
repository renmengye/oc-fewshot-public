DATA_DIR=$1
URL=http://www.cb.uu.se/~gustaf/texture/data/without-rotations-zip/
mkdir -p $DATA_DIR
while read line; do
  fullurl=$URL$line
  echo "Downloading $fullurl"
  wget -P $DATA_DIR $fullurl
done <scripts/uppsala_texture.txt

DIR=$PWD
cd $DATA_DIR
for f in *.zip;
do
  filename=$(basename -- "$f")
  extension="${filename##*.}"
  filename="${filename%.*}";
  echo $filename;
  unzip -o $f -d $filename
done;
rm *.zip
cd $DIR
