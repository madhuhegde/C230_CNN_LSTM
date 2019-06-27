#rm -r data
vid_source_path="data"
target_dir="data"

mkdir $target_dir/images
mkdir $target_dir/images/train
sudo ln -s $target_dir/images/train $target_dir/images/test
sudo ln -s $target_dir/images/train $target_dir/images/eval
mkdir $target_dir/labels
mkdir $target_dir/labels/train
sudo ln -s $target_dir/labels/train $target_dir/labels/test
sudo ln -s $target_dir/labels/train $target_dir/labels/eval


python CNN_LSTM_data_prepare.py