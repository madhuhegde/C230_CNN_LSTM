# C230-CNN-LSTM
CS230-CNN-LSTM: Workflow Recognition from Surgical Videos using LSTM

by Madhu Hegde, Nitin Akkati, Senthil Gunasekaran

## Introduction
The C230-CNN-LSTM repository contains the codes for C230 Spring 2019 project.  The goal of the project is to build a deep neural network algorithm to identify phases of a surgical procedure. It is also planned to integrate the tool localization algorithms into the main algorithm to improve the accuracy. The result of this project will be a deep neural network algorithm that can be trained on surgical videos in an end-to-end manner and provide a high-accuracy phase detection of the surgical procedure. The algorithm is developed  using Cholecystectomy videos first(Cholec80 dataset that captured Gall Bladder Surgery) . The end product is a CNN-LSTM model that can detect surgical phases of Aortic Cannulation (Heart Surgery).  Extending the model to heart surgery depends on the availability of data set. 

Dr M. Hossein suggested this project and helping with datasets.



## Installation
1. Clone the C230-CNN-LSTM repository
    ```shell
    git clone https://github.com/madhuhegde/C230-CNN-LSTM.git
    ```
    
2. Build Environment
- Keras 2.2.4 with Tensorflow 1.13.1 backend

- Python 3.6, OpenCV, ffmpeg

  

## Step by Step Recognition
Most related codes are in `C230-CNN-LSTM/` folder.

1. Download Cholec80 dataset

   The Cholec80  dataset contains 80 videos of cholecystectomy procedures The videos are recorded at 25 fps. All the frames are annotated with 7 defined phases: (1) preparation, (2) calot triangle dissection, (3) clipping and cutting, (4) gallbladder dissection, (5) gallbladder packaging, (6) cleaning and coagulation, and (7) gallbladder retraction. This dataset also contains tool annotations (generated at 1 fps) indicating the presence of seven tools in an image. The seven tools are: grasper, bipolar, hook, scissors, clipper, irrigator, and specimen bag.

2. Data Preprocessing

- The ffmpeg utility is used to split the videos to images. We split the videos at 1 fps for Cholec80 and two videos are used as example in CNN_LSTM_data_prepare.py

  ```shell
  cd C230-CNN-LSTM
  python CNN_LSTM_data_prepare.py
  ```
  
-  The preprocessing generates images corresponding to video1.mp4 clip into images/video1 folder and subsampled ground-truth labels are stored in the text file - labels/video1-label.txt

  

3. Data Pipeline

- The image data is resized from 1920 x 1080 to 224 x 224 to fit VGG16 input dimensions.   The ground truth labels are one-hot coded to (1,7) vectors.

- The list containing paths to entire training input images and labels is created to feed the data generators for the VGG16+LSTM model. Use of list containing path instead of loading entire images helped to reduce the memory requirement of training with video/image data.

- The variable frames_per_clip controls the number of consecutive images grouped together, and corresponding to one surgical phase/class.  The frames_per_clip is set to 25.  The resulting input vector has dimension (num_train_sample, 25, 224, 224, 3)  and label vector has (num_train_sample, 7) 

    

4. Training the network

   The Keras is used for implementation of CNN+LSTM training.

- The baseline model uses  VGG16 (pre-trained on imagenet) as CNN and transfer learning is used.  The number of layers of VGG16 to be trained is a hyper parameter and we have chosen to update last 4 layers (to start with).  
- LSTM network with 80 hidden layers is chosen to start with . The model.summary of LSTM network is given below.  The output is a dense layer with softmax activation for 7 classes  (i.e., phases of the surgery)
- The Nadam optimizer (Adam Optimizer with RMSprop and Nesterov momentum) is used with default values to reduce "categotical_crossentropy"  over 7 classes.
- Function model.fit_generator() is used to train the model on data generated batch-by-batch by a Python generator. Use of generators to feed CNN+LSTM model reduced the memory usage considerably in a batch processing.  It also helps to run the model efficiently as generators run on CPUs in parallel to training the model on GPUs



## Current Status   

- The code is checked into GitHub
- The data pre-processing and pipeline are tested and working as expected. 
- The code runs successfully on AWS EC2 with multiple GPUs.  We have verified training of this CNN+LSTM model on a smaller dataset of 3 videos (4000 images) with TBD accuracy
- All the training is with default hyper-parameters and further optimization is expected



## Next Steps

- Analyze data pre-processing and pipeline to confirm if 1fps and 25 frames per clip is suitable for characterization of surgical videos
- Perform hyper parameter tuning and improve test/validation accuracy using larger data set. Below parameters are likely change
  - Learning Rate and batch_size
  - Number of layers of VGG16 that are updated/frozen
  - Number of hidden layers in LSTM
- Our goal is to achieve an accuracy > 80 % and if this performance goal is not achieved with VGG16 then try better image classification model such as Resnet-50.

- Once 80% accuracy is achieved with CNN+LSTM  integrate the tool localization algorithms into the main algorithm to improve the accuracy

  

## Citation
@article{jin2018sv,  
&nbsp;&nbsp;&nbsp;&nbsp;  title={SV-RCNet: Workflow Recognition From Surgical Videos Using Recurrent Convolutional Network},  
&nbsp;&nbsp;&nbsp;&nbsp;  author={Jin, Yueming and Dou, Qi and Chen, Hao and Yu, Lequan and Qin, Jing and Fu, Chi-Wing and Heng, Pheng-Ann},  
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE transactions on medical imaging},  
&nbsp;&nbsp;&nbsp;&nbsp;  volume={37},  
&nbsp;&nbsp;&nbsp;&nbsp;  number={5},  
&nbsp;&nbsp;&nbsp;&nbsp;  pages={1114--1126},  
&nbsp;&nbsp;&nbsp;&nbsp;  year={2018},  
&nbsp;&nbsp;&nbsp;&nbsp;  publisher={IEEE}  
}

