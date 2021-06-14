# Heart-Rate-Detection-RGB-Videos
Detection of Heart Rate from Frontal Face RGB videos using CNN


[downloaded](https://drive.google.com/file/d/1dyRDSTgg-77ChP9GjwSIdETM4kwZfp6Q/view?usp=sharing) the 68 Facial Landmark model and put it in the current working directory as well as packages folder.

[Link](https://drive.google.com/drive/folders/1zLLrzm2CZ_vesnqFXq2vjAbmSFCZwrac?usp=sharing) to our dataset. Download the dataset and put it in 'stable videos/ISI/Dataset' folder.

Clone the repository.

## Training:
First extract the FFT data from the video files using the data_extraction.ipynb file. Then run the train.ipynb file for training the CNN using the extracted data.

## Inference:
For inference mode or testing the system simply run application.py and input a prerecorded video file. For IP Camera based inference replace the **ipcam_url** variable with your own IP Cam url in *test_model.py*. In case of webcam capture simply pass '0' as the argument in the cv2.VideoCapture() function in *test_model.py*




