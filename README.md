# Vital signs Prediction using computer vision for Diagnostic Application

This repository uses  signal processing and computer vision (face detection,deep learning) technique to predict vital signs from photoplethysmography method. 


# Remote Photoplethysmography

The blood vessels in face when exposed to light shows a pattern of blood volume change which can be tracked using computer vision and signal processing techniques. 


This repository is an approach to calculate different vital signs predicted from blood volume change pattern.

The vital signs predicted in this repository includes:
1) HeartBeat
2) Respiratory Rate
3) SpO2 level
4) Blood Pressure

The vital signs can be used for diagnosis applications.
![overview-heart-rate-and-hrv](https://user-images.githubusercontent.com/65164450/210617201-b3a6574b-1c3f-4db6-9bcd-2aa68f47d417.jpg)



## Installation 
Clone this repository:

```
git clone https://github.com/A0158/Vitalsigns_Prediction.git
cd ./Vitalsigns_Prediction/
```
Install dependencies:
```
conda activate tf-gpu tensorflow-gpu cudatoolkit=10.1
```
This command will install Cuda.
```
pip install requirements.txt
```
## Training 
```
python train.py --exp_name test --data_dir [DATASET_PATH] 
```
## Running Inference to calculate BP, Heart Rate and Respiratory Rate

So we have two methods for running inference:
### 1) Realtime
```
python realtime/predict_realtime.py 
```
It will ask for camera permission. Please allow it to proceed
### 2) Uploading Videos
```
python realtime/upload_video.py --video_path [VIDEO_PATH]

```
![Screenshot (688)](https://user-images.githubusercontent.com/65164450/210620999-1afa0ded-2df4-4a19-8a55-ca3351253298.png)

## Running Inference to calculate SpO2 level
```
python oxygensaturation.py
```


## Steps to dockerize the container
```
docker build -t name:tag .
```

## Run the docker
```
docker run -p 127.0.0.1:80:8080/tcp ubuntu bash
```
If you want to understand about dockers  check [Docker manual](https://docs.docker.com/desktop/)

## Run the model in app using this command:

```
python run app.py
```



