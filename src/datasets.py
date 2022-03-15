
import os
from re import L
from typing import Dict, Tuple
import sys
import logging
import numpy
import regex as re
# import torchaudio.datasets.tedlium as tedlium 
import librosa

from src.Data import tedlium_local as tedlium
import torchaudio
from torch import Tensor
import pandas as pd

from src.utils import get_git_root
from src.Preprocessing.pre_processing import resample_audio

"""Specify path to TEDLIUM directory"""
data_paths = os.path.join(get_git_root(os.getcwd()), 'src' ,'Data')
DATASET_TEDLIUM_PATH = data_paths
DATASET_MLCOMMONS_PATH = data_paths
KEYWORDS_LINK_CSV_PATH = os.path.join(data_paths, "KeywordPerSample", "keywords.csv")
KEYPHRASES_LINK_CSV_PATH = os.path.join(data_paths, "Keyphrases" , "keyphrases.csv")


LABELS_KEYPHRASES_CSV_PATH = os.path.join(data_paths, "Keyphrases" , "labels.csv")


#TODO! Might be better to have a header called keyword_id, in order to take into account the different varations of keywords and phrases inside the same sample
class KeywordsCSVHeaders:
    """
    Represents the fields keywords.csv file
        KEYWORD: The keyword linking the two audio files (sample of a TED audio file and an MSWC recording of that keyword)
        TED_SAMPLE_ID: Represents the sample id of an audio. In other words, it is a unique id that maps to a segment of a TED audio file. 
                    Hence, this is NOT the same as "talk_id", which represents the id of an entire audio file
        TED_DATASET_TYPE: The type of dataset the sample exists in (Train vs Dev vs Test set) 
        MSWC_ID: The id of the keyword recording
    """
    KEYWORD = "Keyword"
    TED_SAMPLE_ID= "TEDLIUM_SampleID"
    TED_DATASET_TYPE = "TEDLIUM_SET"
    MSWC_ID = "MSWC_AudioID"
    CSV_header = [KEYWORD, TED_SAMPLE_ID, TED_DATASET_TYPE, MSWC_ID]


class KeyphrasesCSVHeaders:
    KEYWORD = "Keyword"
    TED_SAMPLE_ID= "TEDLIUM_SampleID"
    TED_DATASET_TYPE = "TEDLIUM_SET"
    MSWC_ID = "MSWC_AudioID"
    KEYWORD_ID = "Word_ID"
    CSV_header = [KEYWORD, TED_SAMPLE_ID, TED_DATASET_TYPE, MSWC_ID, KEYWORD_ID]

class LabelsCSVHeaders:
    """
    Represents the fields labels.csv file
        KEYWORD: The keyword linking the two audio files (sample of a TED audio file and an MSWC recording of that keyword)
        TED_SAMPLE_ID: Represents the sample id of an audio. In other words, it is a unique id that maps to a segment of a TED audio file. 
                    Hence, this is NOT the same as "talk_id", which represents the id of an entire audio file
        TED_DATASET_TYPE: The type of dataset the sample exists in (Train vs Dev vs Test set) 
        MSWC_ID: The id of the keyword recording
    """
    KEYWORD = "Keyword"
    # Keyword_id = "Keyword_id"
    TED_SAMPLE_ID= "TEDLIUM_SampleID"
    TED_DATASET_TYPE = "TEDLIUM_SET"
    TED_TALK_ID = "TED_TALK_ID" 

    MSWC_ID = "MSWC_AudioID"
    START_TIMESTAMP = "start_time"
    END_TIMESTAMP = "end_time"
    CONFIDENCE = "confidence"
    
    CSV_header = [KEYWORD, TED_SAMPLE_ID,TED_TALK_ID,  TED_DATASET_TYPE, MSWC_ID, START_TIMESTAMP, END_TIMESTAMP, CONFIDENCE]


#TODO! Customise for each subset, in speaker-adaptation. Might require changing the metadata
class TEDLIUMCustom(tedlium.TEDLIUM):
    """
    Please have a directory with the TEDLIUM dataset downloaded (release-3).
    Instance Variables: 
        self._path:
        self._filelist:
        self._dict_path:
        self._phoneme_dict: 
    Additional Instance Variables:
        self.train_audio_sets
        self.dev_audio_sets
        self.test_audio_sets
    """
    def __init__(self, root=DATASET_TEDLIUM_PATH, release= "release3", subset=None):
        super().__init__(root, release=release)
     
        path_to_speaker_adaptation = os.path.join(root, tedlium._RELEASE_CONFIGS[release]["folder_in_archive"], "speaker-adaptation")
        train_audio_sets = set(line.strip() for line in open(os.path.join(path_to_speaker_adaptation, "train.lst")))
        dev_audio_sets = set(line.strip() for line in open(os.path.join(path_to_speaker_adaptation, "dev.lst")))
        test_audio_sets = set(line.strip() for line in open(os.path.join(path_to_speaker_adaptation, "test.lst")))

        self.recordings_set_dict = {
            "train": train_audio_sets,
            "dev": dev_audio_sets,
            "test": test_audio_sets
        }



    def __len__(self) -> int:
        """Get number of items.
    
        Returns:
            int: TEDLIUM Dataset Length
        """
        return super().__len__()

    def _load_audio(self, path: str, start_time: float, end_time: float, sample_rate: int = 16000, to_numpy=True) -> [Tensor, int]:
        """
        Returns audio data

        Args:

        Returns:
        """
        waveform, sample_rate =  super()._load_audio(path, start_time, end_time, sample_rate)
        return (waveform.numpy(), sample_rate) if to_numpy else (waveform , sample_rate)

    def __getitem__(self, sampleID: int) -> Dict:

        """Load the n-th sample from the dataset, where n is the audioFileID/fileSampleId
        Please note that filesampleID is different from talk_id returned by the function, which denotes the entire recording instead

        Args:
            AudioFileID (int): The index of the sample to be loaded, which is also termed as the unique ID

        Returns:
            Dictionary: ``(waveform, sample_rate, transcript, talk_id, speaker_id, identifier, start_time, end_time)`` 
            
            """
        fileid, line = self._filelist[sampleID]
        return self._load_tedlium_item(fileid, line, self._path)

    def get_audio_file(self, sampleID:int):
        fileid, line = self._filelist[sampleID]
        return os.path.join(self._path, "sph", fileid)



    def _load_tedlium_item(self, fileid: str, line: int, path: str) -> Dict:
        """Loads a TEDLIUM dataset sample given a file name and corresponding sentence name. Functionality taken from original source code.
        
        ----> Custom function returns start time and end time as well 

        Args:
            fileid (str): File id to identify both text and audio files corresponding to the sample
            line (int): Line identifier for the sample inside the text file
            path (str): Dataset root path

        Returns:
            Dictionary
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, talk_id, speaker_id, identifier, start_time, end_time)``
        """
        transcript_path = os.path.join(path, "stm", fileid)
        with open(transcript_path + ".stm") as f:
            transcript = f.readlines()[line]
            talk_id, _, speaker_id, start_time, end_time, identifier, transcript = transcript.split(" ", 6)

        wave_path = os.path.join(path, "sph", fileid)
        waveform, sample_rate = self._load_audio(wave_path + self._ext_audio, start_time=start_time, end_time=end_time)

        results_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "transcript": transcript, 
            "talk_id": talk_id,
            "speaker_id":speaker_id ,
            "identifier": identifier ,
            "start_time": float(start_time),
            "end_time": float(end_time), 
        }
        return results_dict



class MultiLingualSpokenWordsEnglish():
    MLCOMMONS_FOLDER_NAME = "Multilingual_Spoken_Words"
    AUDIO_DIR_NAME="audio"
    SPLITS_DIR_NAME="splits"
    ALIGNMENTS_DIR_NAME="alignments"

 

    def raise_directory_error(self):
        raise RuntimeError(
            "Please configure the path to the Spoken Keywords Dataset, with the directory name \"{}\", containing the three subfolders:".format(self.MLCOMMONS_FOLDER_NAME) \
            + "\n" + \
            "\"{}\" for audio, \"{}\" for splits directory, and \"{}\" for alignemnts directory".format(self.AUDIO_DIR_NAME,self.SPLITS_DIR_NAME,self.ALIGNMENTS_DIR_NAME)
        )

    #TODO! Accept 4 kinds of values: Train vs test vs Dev vs "all"
    def __init__(self, root=DATASET_MLCOMMONS_PATH, read_splits_file=False, subset="train") -> None:
        """
        Loads the MLCommons MultiLingual dataset (English version).

        read_splits_file is used to generate the keywords csv file
        """
        if self.MLCOMMONS_FOLDER_NAME not in os.listdir(root):
            self.raise_directory_error()
        self._path = os.path.join(root, self.MLCOMMONS_FOLDER_NAME)
        #Initialise the folder names into dictionaries
        self._subfolder_names_dict = {
            "audio" : self.AUDIO_DIR_NAME,
            "splits" : self.SPLITS_DIR_NAME,
            "alignments": self.ALIGNMENTS_DIR_NAME,
        }

        #Check if all three subfolders are in the directory. Exit if they are not all there
        current_subfolders = os.listdir(self._path)
        if not all([subfolder_name in current_subfolders  for subfolder_name in self._subfolder_names_dict.values()]):
            self.raise_directory_error()



        
        #Retrieve the splits csv file from MSWC folder
        if read_splits_file:
            self._path_to_splits = os.path.join(self._path, self._subfolder_names_dict["splits"])
            self.splits_df = pd.read_csv(os.path.join(self._path_to_splits, "en_splits.csv"))
            if subset == "train":
                self.splits_df = self.splits_df[self.splits_df["SET"] == "TRAIN"]
            elif subset == "dev":
                self.splits_df = self.splits_df[self.splits_df["SET"] == "VALID"]
            else:
                self.splits_df = self.splits_df[self.splits_df["SET"] == "TEST"]

            #Extra step to preprocesses words to one form of apostrophe
            self.splits_df["WORD"].replace("`|’", "'", regex=True, inplace=True) 
            #Retrieve the words that have been validated as True, affirming that the spoken audio matches the transcription 
            self.splits_df = self.splits_df[self.splits_df["VALID"] == True]
            #Retrieve the keywords in the dataset
            self.keywords = set(self.splits_df["WORD"].unique())


    def _load_audio(self, path_to_audio, to_numpy=True):
        """Loads audio data from file given file path
        
        Returns:
            waveform: Tensor / np.array
            sample_rate: int
        """
        # waveform, sample_rate =  torchaudio.load(path_to_audio)
        # return (waveform.numpy(), sample_rate) if to_numpy else (waveform , sample_rate)
        waveform, sample_rate = librosa.load(path_to_audio)

        return (waveform, sample_rate) if to_numpy else (waveform , sample_rate)


    def __getitem__(self, MSWC_AudioID) -> Dict:
        """Retrieves sample data from file given Audio ID
        """
        path_to_audio = os.path.join(self._path,self.AUDIO_DIR_NAME ,"en", "clips", MSWC_AudioID)
        waveform, sample_rate= self._load_audio(path_to_audio)
        results_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate 
        }
        return results_dict

#TODO! Create mapping between talk ids and datatype set (i.e not just sample mapping). Use the defined train_audio_sets, dev_audio_sets, test_audio_sets to help. Might be better to implement this in the TEDLIUMCustom instead of here.
class CTRLF_DatasetWrapper:
    """
    Main class wrapper for both TEDLIUM dataset and MSWC dataset. Using the labels csv file, use the functions to retrieve audio samples and their corresponding keywords that was linked to.
        
    Args:
        single_keywords_label: Represents a toggle which defines what types of labels we are dealing with.
            ------------> NOTE: This was added for the time being as handling of multiple keywords may require some changes in the implementation of the code here and elsewhere
    """
    def __init__(self,path_to_labels_csv=LABELS_KEYPHRASES_CSV_PATH, path_to_TED=DATASET_TEDLIUM_PATH, path_to_MSWC=DATASET_MLCOMMONS_PATH, single_keywords_labels=True):
        self._path_to_TED = path_to_TED
        self._path_to_MSWC = path_to_MSWC
        self.single_keywords_labels = single_keywords_labels
        #Initialise keyword dataframe
        self.labels_df = pd.read_csv(path_to_labels_csv)
        self.audio_keywords_dataset_dict = {
            "train": self.labels_df[self.labels_df["TEDLIUM_SET"] == "train"],
            "dev": self.labels_df[self.labels_df["TEDLIUM_SET"] == "dev"],
            "test": self.labels_df[self.labels_df["TEDLIUM_SET"] == "test"]
        }

        #Initialise Ted talk dataset
        self.TED = TEDLIUMCustom(root=path_to_TED,release="release3")

        #Initialise Keyword dataset
        self.MSWC = MultiLingualSpokenWordsEnglish(root=path_to_MSWC)


    
    def get(self, TEDSample_id: int):
        """
        Given Ted Sample ID and the dataset type, return three separate corresponding dictionaries.
        Returns:
            TED_results_dict:
                { 
                    "waveform": audio data of the Ted talk sample as type Tensor,
                    "sample_rate": sample rate as type int ,
                    "transcript": transcript string as type str, 
                    "talk_id": talk id (of the entire audio file) as str,
                    "speaker_id": speaker id as str (Not needed),
                    "identifier": (Not needed),
                    "start_time": start time of the audio sample in seconds,
                    "end_time": end time of the audio sample in seconds, 
                }
            MSWC_results_dict:
                {
                    "waveform": audio data of the keyword recording
                    "sample_rate": sample rate of the keyword recording
                }
            label_dict = {
                "keyword": keyword,
                "start_time": estimated start time stamp of the label,
                "end_time": estimated end time stamp of the label, 
            }
        """
        TED_results_dict = self.TED.__getitem__(TEDSample_id)

        TEDSample_id = str(TEDSample_id) #TODO: Return pandas in appropriate form
        MSWC_audio_ids = self.labels_df[self.labels_df[LabelsCSVHeaders.TED_SAMPLE_ID] == int(TEDSample_id)]
        if len(MSWC_audio_ids) == 0:
            print("*" * 80)
            print("NOT FOUND: \nSample TED Audio ID {} does not exist in the csv file".format(TEDSample_id))
            print("If you think it should exist, please check the data types you are comparing with (i.e str vs int) and the csv file itself")
            print("*" * 80)
        MSWC_results_dict = None
        if self.single_keywords_labels and len(MSWC_audio_ids) != 0:
            MSWC_results_dict =  self.MSWC.__getitem__(MSWC_audio_ids[LabelsCSVHeaders.MSWC_ID].iloc[0])
        

        #Resample Audio files into same sampling rate
        TED_results_dict, MSWC_results_dict = self.resample_both_audio_files(TED_results_dict, MSWC_results_dict)
        row = self.labels_df[self.labels_df[LabelsCSVHeaders.TED_SAMPLE_ID] == int(TEDSample_id)]
        label_dict = {
            "keyword": row[LabelsCSVHeaders.KEYWORD].iloc[0],
            "start_time": row[LabelsCSVHeaders.START_TIMESTAMP].iloc[0],
            "end_time": row[LabelsCSVHeaders.END_TIMESTAMP].iloc[0], 
        }
        return TED_results_dict, MSWC_results_dict, label_dict 

    # Helper function: Preprocessing step to ensure both audio files are on the same sampling rate
    def resample_both_audio_files(self, TED_results_dict, MSWC_results_dict, target_rate=16000):
        TED_results_dict["waveform"] = resample_audio(TED_results_dict["waveform"], TED_results_dict["sample_rate"], target_rate=target_rate)
        TED_results_dict["sample_rate"] = target_rate
        MSWC_results_dict["waveform"] = resample_audio(MSWC_results_dict["waveform"], MSWC_results_dict["sample_rate"], target_rate=target_rate)
        MSWC_results_dict["sample_rate"] = target_rate
        return TED_results_dict, MSWC_results_dict
    
    #TODO! Remove sorting. Better to Sort beforehand,  instead of here... for faster iteration
    #Retrieve all the available "samples" of one specific audio file
    def get_samples_given_talk_id(self, TED_talk_id, sort=False):
        samples_df = self.labels_df[self.labels_df[LabelsCSVHeaders.TED_TALK_ID] == int(TED_talk_id)]
        if sort:
            samples_df.sort_values(by=['col1'], inplace=True)
        return samples_df

if __name__== "__main__":
    ####### Testing CTRLF_DatasetWrapper
    print("-"*20)  
    print("CTRL_F Wrapper") 

    x= CTRLF_DatasetWrapper()
    Ted_dict, MSWC_dict, label_dict= x.get(0)
    print(Ted_dict)
    print(MSWC_dict)
    print(label_dict)

    ####### Testing TEDLIUM
    print("-"*20)  
    print("Tedlium") 

    y = TEDLIUMCustom()
    print(y.__len__())
    print(y.__getitem__(1))


    ####### Testing MultiLingualSpokenWordsEnglish 
    print("-"*20)  
    print("Keyword Dataset") 

    # z = MultiLingualSpokenWordsEnglish(read_splits_file=True)
    z= MultiLingualSpokenWordsEnglish(read_splits_file=False)

    print(z.__getitem__("aachen/common_voice_en_20127845.opus"))
