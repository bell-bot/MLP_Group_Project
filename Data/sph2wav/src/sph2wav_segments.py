from src.datasets import CTRLF_DatasetWrapper 
from tqdm import tqdm
import scipy.io.wavfile
import os
import tensorflow as tf
import numpy as np
path = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/Data/TEDLIUM_release-3/data/sph/'  # Path of folder containing .sph files
out_path = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/Data/TEDLIUM_release-3/data/wav/'

#NOTE: Writes only wav files that have at least one label timestamp in that audio segment
x = CTRLF_DatasetWrapper()
for i in sorted(list(x.TED_sampleids_in_labels_set)):
    print(i)
    TED_sample_dict = x.TED.__getitem__(i)
    Ted_talk_id = TED_sample_dict["talk_id"]
    ofile = str(i)+"_" +Ted_talk_id +".wav" 
    # output_signal = tf.audio.encode_wav(TED_sample_dict["waveform"], TED_sample_dict["sample_rate"])
    scipy.io.wavfile.write(out_path + ofile, TED_sample_dict["sample_rate"], data = TED_sample_dict["waveform"][0].astype(np.int16)) #
    # tf.io.write_file(out_path + ofile , output_signal)


