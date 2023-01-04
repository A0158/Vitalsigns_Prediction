import tensorflow as tf
import numpy as np
import scipy.io
import os
import cv2
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
#from testcv.py import Plotter
from inference import preprocess_raw_video, detrend

def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './mtts_can.hdf5'
    batch_size = args.batch_size
    fs = args.sampling_rate
    #sample_data_path = args.video_path
    
    vidObj = cv2.VideoCapture(0)
    def make_1080p():
      vidObj.set(3, 1920)
      vidObj.set(4, 1080)

    def make_720p():
      vidObj.set(3, 1280)
      vidObj.set(4, 720)

    def make_480p():
      vidObj.set(3, 640)
      vidObj.set(4, 480)

    def change_res(width, height):
      vidObj.set(3, width)
      vidObj.set(4, height)
    make_480p()
    while(vidObj.isOpened()):
        #totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
        #Xsub = np.zeros((totalFrames, 36, 36, 3))

        height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
        success, img = vidObj.read()
        dims = img.shape
        
        cv2.imshow('output',success)

    #make_1080p()
    #change_res(4000,2000)
        
        print("Orignal Height", height)
        print("Original width", width)
        dXsub = preprocess_raw_video(vidObj)
        print('dXsub shape', dXsub.shape)

        dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
        dXsub = dXsub[:dXsub_len, :, :, :]

        model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
        model.load_weights(model_checkpoint)

        yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

        pulse_pred = yptest[0]
        pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
        signal=pulse_pred
        signal_size = len(signal)
        signal = signal.flatten()
        fft_data = np.fft.rfft(signal) # FFT
        fft_data = np.abs(fft_data)
        freq = np.fft.rfftfreq(signal_size, 1./30) # Frequency data   
        bps_freq=60.0*freq
        max_index = np.argmax(fft_data)
        HR =  bps_freq[max_index]
        print(HR)

        resp_pred = yptest[1]
        resp_pred = detrend(np.cumsum(resp_pred), 100)
        [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))
    
        #p=Plotter(400,200)
        #p.plot()
    ########## Plot ##################
        plt.subplot(211)
        plt.plot(pulse_pred,color="red")
        plt.title('Pulse Prediction')
        plt.subplot(212)
        plt.plot(resp_pred,color="blue")
        plt.title('Resp Prediction')
        plt.show()
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    vidObj.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    args = parser.parse_args()

    predict_vitals(args)

