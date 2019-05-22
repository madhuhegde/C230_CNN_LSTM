#rm -r data
vid_source_path="../../C230_CNN_LSTM/data"
mkdir data
mkdir data/phase_annotations
mkdir data/videos
mkdir data/videos/train
mkdir data/videos/test
mkdir data/images
mkdir data/images/train
mkdir data/images/test
mkdir data/labels
mkdir data/labels/train
mkdir data/labels/test


declare -a trainvideoindexes=("02" "03" "04" "05" "06" "10" "11" "12" "13" "14" "15" "17" "18" "21" "22" "24" "25" "26" "29" "32" "36" "37" "41" "43" "48" "49" "50" "51" "53")

declare -a testvideoindexes=("01" "16" "40" "47")


for i in "${trainvideoindexes[@]}"
do
    echo "copy $vid_source_path/video$i* to data/videos/train/"
    cp $vid_source_path/videos/video$i* data/videos/train/
done

for i in "${testvideoindexes[@]}"
do
    echo "copy $vid_source_path/video$i* to data/videos/test/"
    cp $vid_source_path/videos/video$i* data/videos/test/ 
done

cp $vid_source_path/phase_annotations/* data/phase_annotations/

python CNN_LSTM_data_prepare.py
