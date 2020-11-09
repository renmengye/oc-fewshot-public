source setup_environ.sh
URL=https://github.com/brendenlake/omniglot/archive/master.zip
FOLDER=roaming-omniglot
wget -P $DATA_DIR $URL

DIR=$PWD
cd $DATA_DIR
unzip -o master.zip -d $FOLDER
rm master.zip
cd $FOLDER
mv omniglot-master/* .
rm -rf omniglot-master
rm -rf matlab
mkdir images_all
unzip -o python/images_background.zip -d images_all
unzip -o python/images_evaluation.zip -d images_all
cd images_all
mv images_background/* .
mv images_evaluation/* .
rmdir images_background
rmdir images_evaluation
cd ..
rm -rf python
rm *.png
rm *.jpg
rm README.md
cd $DIR