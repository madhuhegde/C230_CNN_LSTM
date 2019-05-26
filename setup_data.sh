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

#declare -a trainvideoindexes=("02" "03" "04" "05" "06" "10" "11" "12" "13" "14" "15" "17" "18" "21" "22" "24" "25" "26" "29" "32" "36" "37" "41" "43" "48" "49" "50" "51" "53")
declare -a trainvideoindexes=("02" "03" "04" "05" "06" "10" "11" "12" "13" "14")

#declare -a testvideoindexes=("01" "16" "40" "47")
declare -a testvideoindexes=("16")

declare -a evalvideoindexes=("77" "78" "79" "80")



for i in "${trainvideoindexes[@]}"
do
    echo "copy $vid_source_path/video$i* to $target_dir/videos/train/"
    #cp $vid_source_path/videos/video$i* $target_dir/videos/train/
done

for i in "${testvideoindexes[@]}"
do
    echo "copy $vid_source_path/video$i* to $target_dir/videos/test/"
    #cp $vid_source_path/videos/video$i* $target_dir/videos/test/ 
done

for i in "${evalvideoindexes[@]}"
do
    echo "copy $vid_source_path/video$i* to $target_dir/videos/eval/"
    cp $vid_source_path/videos/video$i* $target_dir/videos/eval/
done

cp $vid_source_path/phase_annotations/* $target_dir/phase_annotations/

#python CNN_LSTM_data_prepare.py
