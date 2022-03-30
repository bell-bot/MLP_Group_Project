import pandas as pd
import numpy as np
from scipy.io.wavfile import read, write
from src.datasets import CTRLF_DatasetWrapper, TEDLIUMCustom
from src.constants import ADVERSERIAL_DATASET_PATH, LABELS_WITH_ADVERSERIAL
from src.Preprocessing.pre_processing import read_WAV_file

def train_validate_test_split(df, train_percent=.8, valid_percent=.1, seed=None, deepspeech_format=True):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)


    num_of_samples = df.shape[0]
    num_of_training = int(num_of_samples*train_percent)
    num_of_valid =  num_of_training + int(num_of_samples*valid_percent)
    
    train = df.iloc[perm[:num_of_training]]
    valid = df.iloc[perm[num_of_training:num_of_valid]]
    test = df.iloc[perm[num_of_valid:]]
    
    
    if deepspeech_format:
        train = parse_adverserial_to_deepspeech_format(train)
        valid = parse_adverserial_to_deepspeech_format(valid)
        test = parse_adverserial_to_deepspeech_format(test)

        
    return train, valid, test

def parse_adverserial_to_deepspeech_format(split_df, preprocess=False):
    x = TEDLIUMCustom()
    ids = split_df.TEDLIUM_SampleID.values
    split_df.reset_index(inplace=True)
    rows = []
    for i in range(split_df.shape[0]):
        split_row = split_df.iloc[i]
        TED_sample_dict = x.__getitem__(split_row.TEDLIUM_SampleID)
        adv_filename = ADVERSERIAL_DATASET_PATH + "/" + split_row.TEDLIUM_SET + ".wav"
        
        _, data = read(adv_filename) # fs is the sampling frequency and data is the read in audio file

        final_wav_name = "Data/adversarial/" + split_row["TEDLIUM_SET"] + ".wav"
        wav_size = len(data)
        transcript = TED_sample_dict["transcript"]
        rows.append([final_wav_name,wav_size,transcript ])
    
    final_df = pd.DataFrame(data= rows, columns={"wav_filename", "wav_filesize", "transcript"})
    return final_df

def save(df, filepath):
    df.to_csv(filepath)
if __name__=="__main__":
    
    adv_labels = pd.read_csv(LABELS_WITH_ADVERSERIAL)
    train, valid, test = train_validate_test_split(adv_labels)
    
