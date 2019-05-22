import io
import os
import subprocess
import glob
base_dir = "/Users/madhuhegde/work/cs230/data/cholec_mini_data/"
phase_gt_dir = base_dir+"phase_annotations/"
video_base_dir = base_dir+"videos/"
image_base_dir = base_dir+"images/"
label_base_dir = base_dir+"labels/"
import pdb


#num_videos = 1


def extract_images(video,output):
    command = "ffmpeg -i {video} -r 1  -q:v 2 -f image2 {output}".format(video=video, output=output)
    #command = "echo  {video}  {output}".format(video=video, output=output)
    subprocess.call(command,shell=True)
    return

def resize_images(image_path):
    image_files = glob.glob(image_path+"*.jpg")
    for image_file in image_files:
       command = "ffmpeg -y -i {input} -vf scale=250:250 {output}".format(input=image_file, output=image_file)
       subprocess.call(command,shell=True)
       #print(image_file)
    
    
def generate_gt_data(in_file, fps):
  #generate 
  out_list = []
  step = int(25/fps)
  with open(in_file, 'r') as handle:
    out_list.append(handle.readline())
    for lineno, line in enumerate(handle):
        if lineno % step == 0:
            templine = line.split('\t')
            templine = str(int(int(templine[0])/step))+'\t'+templine[1]
            out_list.append(templine)
                      
  return(out_list)
    
  
  
#pdb.set_trace() 
def process_videos(video_dir, image_dir, label_dir, num_videos=1):

  video_files = glob.glob(video_dir+"*.mp4")
  
  for video_num, video_file in enumerate(video_files):
     file_name = video_file.split('/')[-1]
     file_name = os.path.splitext(file_name)[0]
     
     image_folder_cmd = "mkdir " + image_dir+file_name
     os.system(image_folder_cmd)
     
     image_folder_name = image_dir+file_name+'/'+file_name+'-%d.jpg'
     print(file_name, image_folder_name+'\n')
     
     #extract images from videos
     
     extract_images(video_file, image_folder_name)
     
     #resize images to 250 x 250. Currently hardcoded to 250 x 250.
     #existing images are overwritten
     #resize_images(image_dir+file_name)
     
     gt_file_name = phase_gt_dir+file_name+"-phase.txt"
     
     fps = 1   # make sure it matches ffmpeg argument
     gt_list = generate_gt_data(gt_file_name, fps)
     #print(gt_list)
     gt_label_file = label_dir+file_name+"-label.txt"
     #print(gt_label_file)
     with open(gt_label_file, 'w') as handle:
        handle.writelines(gt_list)
     
     if(video_num >= num_videos-1):
       break       
       
  return video_num     
  
  
if __name__ == "__main__":
     num_train_videos = 16
     num_test_videos = 2
     train_video_path = video_base_dir+"train/"
     test_video_path = video_base_dir+"test/"
     train_images_path = image_base_dir+"train/"
     test_images_path = image_base_dir+"test/"
     train_labels_path = label_base_dir+"train/"
     test_labels_path = label_base_dir+"test/"
     
     
     train_num = process_videos(train_video_path, train_images_path, train_labels_path, num_train_videos)
     test_num = process_videos(test_video_path, test_images_path, test_labels_path, num_test_videos)
     
     
  
