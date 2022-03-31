import pandas as pd
import librosa
import os
import numpy as np
import torch
from torch.nn import L1Loss, MSELoss
from enum import Enum
from tqdm import tqdm
import gc
import ctypes

from src.utils import get_git_root
from src.datasets import CTRLF_DatasetWrapper

def features_extractor(audio): 
    mfccs_features = librosa.feature.mfcc(y=audio, n_mfcc=13)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

def features_extractor_windowed(frames):
    return np.apply_along_axis(features_extractor, 1, frames)

def window_split(frame_len, hop_len,data, print_frame = False):
    frames = librosa.util.frame(data, frame_length=frame_len, hop_length=hop_len)
    #windowed_frames = np.hanning(frame_len).reshape(-1, 1) * frames
        
    return windowed_frames
    

def mae_loss(x, y):
    loss = np.square(np.subtract(x, y)).mean()
    return loss

def rmse_loss(x,y):
    loss = MSELoss()
    return torch.reshape(torch.sqrt(loss(x,y)), shape=(1,1))

class LossType(Enum):
    MAE = "mae"
    RMSE = "rmse"


def compare_window(keyword, window, loss_type=LossType.MAE):

    if loss_type == LossType.MAE:
        return mae_loss(keyword, window)

    elif loss_type == LossType.RMSE:
        return rmse_loss(keyword, window)

def match_audio(keyword, sliding_windows, loss_type=LossType.MAE):
    #results = []
    #for i in range(len(sliding_windows)):
    #    t = sliding_windows[i]
    #    loss = compare_window(keyword, t, loss_type=loss_type)
    #    results.append(loss)
    #return torch.cat(results, dim=0)
    # Duplicate keyword s.t. we have the same number of rows as sliding_windows
    keyword_dup = np.tile(keyword, sliding_windows.shape)
    return mae_loss(keyword_dup, sliding_windows)

def match_timestamps(start, end, actual_start, actual_end):
    start_valid = start - 0.1 < actual_start and start + 0.1 > actual_start
    end_valid = end - 0.1 < actual_end and end + 0.1 > actual_end
    return start_valid and end_valid

# Initialise the CTRL-wrapper for datasets.py
x = CTRLF_DatasetWrapper()

# For instance in the data set wrapper, try to predict the keyword
# accuracy stores an array of 0 and 1, where 0 indicates that the model prediction was wrong, 
# and 1 indicates the the model prediction was correct
accuracy = []

# idx keeps track of the ted talk id
idx = 0

samples_to_write= sorted(list(x.TED_sampleids_in_labels_set))

for i in tqdm(range(len(samples_to_write))):
    idx = samples_to_write[i]
    # Retrieve the sample at the current index
    sample = x.get(idx)

    # We require the ted audio waveform, keyword waveform, audio start and end time, and keyword start and 
    # end time
    ted_waveform = sample["TED_waveform"][0]

    keywords = sample["MSWC_audio_waveform"].tolist()
    ted_start_time = sample["TED_start_time"][0]
    ted_end_time = sample["TED_end_time"][0]
    ted_length = ted_end_time-ted_start_time
    ted_sample_rate = sample["TED_sample_rate"][0]
    keyword_sample_rate = sample["MSWC_sample_rate"].tolist()
    keyword_start_time = sample["keyword_start_time"].tolist()
    keyword_end_time = sample["keyword_end_time"].tolist()

    # Iterate through all keywords:

    for (keyword, keyword_start, keyword_end, keyword_sr) in zip(keywords, keyword_start_time, keyword_end_time, keyword_sample_rate):

        # Length of the window has to be the length of the keyword
        window_len = len(keyword)

        # Window the ted waveform
        frames, windowed_frames = window_split(window_len, 1, ted_waveform)

        # Convert the frames and keyword waveform to MFCC
        ted_mfcc = features_extractor_windowed(windowed_frames[0].T)

        # Convertthe keyword to mfcc
        keyword_mfcc = features_extractor(keyword)
        # For each window, compute the mse of the window and the keyword
        windowed_mse = compare_window(keyword_mfcc, ted_mfcc)

        # Indentify the window with the least mse
        
        least_mse = np.argmin(windowed_mse)

        coef_ted = ted_length/ted_waveform.shape[1]
        coef_keyword = (keyword_end-keyword_start)

        # Compute the start and end timestamp
        start_timestamp = ted_start_time + least_mse*coef_ted
        end_timestamp = start_timestamp + coef_keyword

        accuracy.append(match_timestamps(start_timestamp, end_timestamp, keyword_start, keyword_end))
        print(f"Sample {idx} done.\n")
        del keyword_mfcc
        del ted_mfcc
        del windowed_mse
        del frames
        del windowed_frames
        del ted_waveform
        gc.collect()
        

final_accuracy = np.sum(accuracy)/len(accuracy)
acc = open("final_acc.txt", "w")
acc.write(str(final_accuracy))
acc.close()
