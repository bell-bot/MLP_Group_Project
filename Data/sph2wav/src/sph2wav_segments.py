from src.datasets import CTRLF_DatasetWrapper 
from tqdm import tqdm
import scipy.io.wavfile
import os
import tensorflow as tf
import numpy as np
path = '/home/wassim_jabrane/MLP_Group_Project/Data/TEDLIUM_release-3/data/sph/'  # Path of folder containing .sph files
out_path = '/home/wassim_jabrane/MLP_Group_Project/Data/TEDLIUM_release-3/data/wav/'

#NOTE: Writes only wav files that have at least one label timestamp in that audio segment
x = CTRLF_DatasetWrapper()
samples_to_write= sorted(list(x.TED_sampleids_in_labels_set))

for i in tqdm(range(len(samples_to_write))):
    sample = samples_to_write[i]
    TED_sample_dict = x.TED.__getitem__(sample)
    Ted_talk_id = TED_sample_dict["talk_id"]
    ofile = str(sample)+"_" +Ted_talk_id +".wav" 
    # output_signal = tf.audio.encode_wav(TED_sample_dict["waveform"], TED_sample_dict["sample_rate"])
    # scipy.io.wavfile.write(out_path + ofile, TED_sample_dict["sample_rate"], data = TED_sample_dict["waveform"][0].astype(np.int16)) #
    ted_waveform = TED_sample_dict["waveform"]
    ted_waveform = ted_waveform.reshape(ted_waveform.shape[1],1)
    encode = tf.audio.encode_wav(
    ted_waveform, TED_sample_dict["sample_rate"], name=None
    )
    tf.io.write_file(
    out_path + ofile, encode, name=None
    )
    # tf.io.write_file(out_path + ofile , output_signal)


