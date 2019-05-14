import io
import os
import subprocess
import glob
base_dir = "data/"
phase_gt_dir = base_dir+"phase_annotations/"
video_dir = base_dir+"videos/"
image_dir = base_dir+"images/"
label_dir = base_dir+"labels/"
import pdb
video_files = glob.glob(video_dir+"*.mp4")

num_videos = 25 


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
    
    
def generate_gt_data(in_file):
  #generate 
  out_list = []
  step = 25
  with open(in_file, 'r') as handle:
    out_list.append(handle.readline())
    for lineno, line in enumerate(handle):
        if lineno % step == 0:
            templine = line.split('\t')
            templine = str(int(int(templine[0])/step))+'\t'+templine[1]
            out_list.append(templine)
                      
  return(out_list)
    
  
#pdb.set_trace() 
for video_num, file in enumerate(video_files):
     file_name = file.split('/')[-1]
     file_name = os.path.splitext(file_name)[0]
     image_name = image_dir+file_name+'/'+file_name+'-%d.jpg'
     
     image_folder_cmd = "mkdir " + image_dir+file_name
     os.system(image_folder_cmd)
     print(file_name, file, image_name+'\n')
     
     #extract images from videos
     
     extract_images(file, image_name)
     
     #resize images to 250 x 250. Currently hardcoded to 250 x 250.
     #existing images are overwritten
     #resize_images(image_dir+file_name)
     
     gt_file_name = phase_gt_dir+file_name+"-phase.txt"
     gt_list = generate_gt_data(gt_file_name)
     #print(gt_list)
     gt_label_file = label_dir+file_name+"-label.txt"
     #print(gt_label_file)
     with open(gt_label_file, 'w') as handle:
        handle.writelines(gt_list)
     
     if(video_num >= num_videos-1):
       break       
