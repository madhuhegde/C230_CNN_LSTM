#rm -r data
vid_source_path="data"
target_dir="data"
mkdir $target_dir
mkdir $target_dir/phase_annotations
mkdir $target_dir/videos
mkdir $target_dir/videos/train
mkdir $target_dir/videos/test
mkdir $target_dir/videos/eval
mkdir $target_dir/images
mkdir $target_dir/images/train
mkdir $target_dir/images/test
mkdir $target_dir/images/eval
mkdir $target_dir/labels
mkdir $target_dir/labels/train
mkdir $target_dir/labels/test
mkdir $target_dir/labels/eval

declare -a trainvideoindexes=("01" "02" "04" "06" "07" "08" "09" "15" "19" "24" "25" "26" "27" "28" "29" "30" "31" "32" "34" "35" "37" "38" "39" "40" "41" "42" "46")

declare -a testvideoindexes=("03" "05" "20" "21" "23" "33" "36" "43" "44" "45" "47" "48" "55" "57")

declare -a evalvideoindexes=("60" "63" "64" "68" "69" "71" "72" "73" "74" "75" "77" "78" "79" "80")


for i in "${trainvideoindexes[@]}"
do
    echo "copy $vid_source_path/video$i* to $target_dir/videos/train/"
    cp $vid_source_path/videos/video$i* $target_dir/videos/train/
done

for i in "${testvideoindexes[@]}"
do
    echo "copy $vid_source_path/video$i* to $target_dir/videos/test/"
    cp $vid_source_path/videos/video$i* $target_dir/videos/test/ 
done

for i in "${evalvideoindexes[@]}"
do
    echo "copy $vid_source_path/video$i* to $target_dir/videos/eval/"
    cp $vid_source_path/videos/video$i* $target_dir/videos/eval/
done

cp $vid_source_path/phase_annotations/* $target_dir/phase_annotations/

python CNN_LSTM_data_prepare.py