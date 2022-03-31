
import os
import pathlib
from re import L
from typing import Dict, Tuple
import sys
import logging
import numpy as np
import regex as re
# import torchaudio.datasets.tedlium as tedlium 
import librosa

from Data import tedlium_local as tedlium
import torchaudio
from torch import Tensor
import pandas as pd

from src.utils import get_git_root
from src.Preprocessing.pre_processing import resample_audio

from random import randint

from src.constants import DATASET_TEDLIUM_PATH, DATASET_MLCOMMONS_PATH, LabelsCSVHeaders, LABELS_KEYPHRASES_CSV_PATH

############# --------- DATASETS --------------################

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

    MSWC_EN_AUDIO_FOLDER = os.path.join(DATASET_MLCOMMONS_PATH, \
                    MLCOMMONS_FOLDER_NAME, \
                    AUDIO_DIR_NAME, \
                    "en", \
                    "clips")
 

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
            self.load_splits_df()
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
    
    def load_splits_df(self):
        self._path_to_splits = os.path.join(self._path, self._subfolder_names_dict["splits"])
        self.splits_df = pd.read_csv(os.path.join(self._path_to_splits, "en_splits.csv"))



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
            "sample_rate": sample_rate ,
            "MSWC_AudioID": MSWC_AudioID
        }
        return results_dict





#TODO! Create mapping between talk ids and datatype set (i.e not just sample mapping). Use the defined train_audio_sets, dev_audio_sets, test_audio_sets to help. Might be better to implement this in the TEDLIUMCustom instead of here.
class CTRLF_DatasetWrapper:
    COLS_OUTPUT= ['TED_waveform', 'TED_sample_rate', 'TED_transcript', 'TED_talk_id', 'TED_start_time', 'TED_end_time', 'MSWC_audio_waveform', 'MSWC_sample_rate', 'MSWC_ID', 'keyword', 'keyword_start_time', 'keyword_end_time', 'confidence']
    COLS_OUTPUT_TED_SIMPLIFIED = ["TED_waveform", "TED_transcript"] #Used for the keras ASR model
    """
    Main class wrapper for both TEDLIUM dataset and MSWC dataset. Using the labels csv file, use the functions to retrieve audio samples and their corresponding keywords that was linked to.
        
    Args:
        single_keywords_label: Represents a toggle which defines what types of labels we are dealing with.
            ------------> NOTE: This was added for the time being as handling of multiple keywords may require some changes in the implementation of the code here and elsewhere
    """
    def __init__(self,path_to_labels_csv=LABELS_KEYPHRASES_CSV_PATH, path_to_TED=DATASET_TEDLIUM_PATH, path_to_MSWC=DATASET_MLCOMMONS_PATH, single_keywords_labels=True, label_mswc_id_error_handling=True):
        self._path_to_TED = path_to_TED
        self._path_to_MSWC = path_to_MSWC
        self.single_keywords_labels = single_keywords_labels
        #Initialise keyword dataframe
        self.labels_df = pd.read_csv(path_to_labels_csv)
        #Initialise Ted talk dataset
        self.TED = TEDLIUMCustom(root=path_to_TED,release="release3")

        #Initialise Keyword dataset
        self.MSWC = MultiLingualSpokenWordsEnglish(root=path_to_MSWC)
        
        #Store the TED sample ids found in set()
        self.TED_sampleids_in_labels_set = set(self.labels_df[LabelsCSVHeaders.TED_SAMPLE_ID].unique())

    #NOTE: THE TEDLIUM_SET IS DEPRECATED, will need to be handled as well in the future. It is currently not used
    def get(self, TEDSample_id: int, sampling_rate=16000, label_mswc_id_error_handling=True):
        """
        Given Ted Sample ID and the dataset type, return three separate corresponding dictionaries.
        Returns: DataFrame
            Headers: 
            ['TED_waveform', 'TED_sample_rate', 'TED_transcript', 'TED_talk_id', 'TED_start_time', 'TED_end_time', 'MSWC_audio_waveform', 'MSWC_sample_rate', 'MSWC_ID', 'keyword', 'keyword_start_time', 'keyword_end_time', 'confidence']

        """
        
        TED_results_dict = self.TED.__getitem__(TEDSample_id)

        TEDSample_id = str(TEDSample_id) #TODO: Return pandas in appropriate form
        label_rows = self.labels_df[self.labels_df[LabelsCSVHeaders.TED_SAMPLE_ID] == int(TEDSample_id)].reset_index()
        
        
        if len(label_rows) == 0:
            print("*" * 80)
            print("NOT FOUND: \nSample TED Audio ID {} does not exist in the csv file".format(TEDSample_id))
            print("If you think it should exist, please check the data types you are comparing with (i.e str vs int) and the csv file itself")
            print("*" * 80)
        output_rows = []
        for i in range(0,len(label_rows)):
            #Labels.csv error, in case keyword audio file is not linked to the right recording.
            if label_mswc_id_error_handling:
                #Get the keyword in the current row
                keyword = label_rows[LabelsCSVHeaders.KEYWORD][i]
                #find the folder in MSWC dataset
                currentDirectory = (self.MSWC.MSWC_EN_AUDIO_FOLDER + "/" +  keyword)

                #Find a random recording\
                audio_files  = (os.listdir(currentDirectory))

                random_number = randint(0,len(audio_files)-1)

                corrected_keyword_ID= os.path.join(keyword, audio_files[random_number])
                label_rows[LabelsCSVHeaders.MSWC_ID][i] =  corrected_keyword_ID
            

            MSWC_results_dict =  self.MSWC.__getitem__(label_rows[LabelsCSVHeaders.MSWC_ID].iloc[i])
            #Resample MSWC  to the sampling rate of TED
            MSWC_results_dict["waveform"] = resample_audio(MSWC_results_dict["waveform"], MSWC_results_dict["sample_rate"], target_rate=TED_results_dict["sample_rate"])
            MSWC_results_dict["sample_rate"] = TED_results_dict["sample_rate"]
            # TED_results_dict, MSWC_results_dict = self.resample_both_audio_files(TED_results_dict, MSWC_results_dict)

            #Create new row
            new_row = [  \
                TED_results_dict["waveform"], \
                TED_results_dict["sample_rate"],\
                TED_results_dict["transcript"],\
                TED_results_dict["talk_id"],\
                TED_results_dict["start_time"],\
                TED_results_dict["end_time"],\
                MSWC_results_dict["waveform"],\
                MSWC_results_dict["sample_rate"],\
                label_rows.iloc[i][LabelsCSVHeaders.MSWC_ID],\
                label_rows.iloc[i][LabelsCSVHeaders.KEYWORD],\
                label_rows.iloc[i][LabelsCSVHeaders.START_TIMESTAMP],\
                label_rows.iloc[i][LabelsCSVHeaders.END_TIMESTAMP], \
                label_rows.iloc[i][LabelsCSVHeaders.CONFIDENCE], \
            ]
            


            output_rows.append(new_row)

        output_df = pd.DataFrame(data=output_rows, columns=self.COLS_OUTPUT)
        return output_df
    
    
    #get function that returns minimal data needed, 1D array/
    def get_ted_talk_id_and_transcript(self,TEDSample_id: int):
        if TEDSample_id not in self.TED_sampleids_in_labels_set:
            print("*" * 80)
            print("NOT FOUND: \nSample TED Audio ID {} does not exist in the csv file".format(TEDSample_id))
            print("If you think it should exist, please check the data types you are comparing with (i.e str vs int) and the csv file itself")
            print("*" * 80)
            return []
        
        
        fileid, line = self.TED._filelist[TEDSample_id]
        transcript_path = os.path.join(self.TED._path, "stm", fileid)
        with open(transcript_path + ".stm") as f:
            transcript = f.readlines()[line]
            talk_id, _, speaker_id, start_time, end_time, identifier, transcript = transcript.split(" ", 6)


        #Create new row
        output_row = [talk_id, transcript]

        return output_row #1D array

    #TODO: Return more results like speaker_id, etc..
    def get_verbose(TEDSample_id: int, sampling_rate = 16000):
        pass

    # Helper function: Preprocessing step to ensure both audio files are on the same sampling rate
    def resample_both_audio_files(self, TED_results_dict, MSWC_results_dict, target_rate=16000):
        TED_results_dict["waveform"] = resample_audio(TED_results_dict["waveform"], TED_results_dict["sample_rate"], target_rate=target_rate)
        TED_results_dict["sample_rate"] = target_rate
        MSWC_results_dict["waveform"] = resample_audio(MSWC_results_dict["waveform"], MSWC_results_dict["sample_rate"], target_rate=target_rate)
        MSWC_results_dict["sample_rate"] = target_rate
        return TED_results_dict, MSWC_results_dict
    
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
    
    print("Dataframe Results")
    x= CTRLF_DatasetWrapper(path_to_labels_csv = LABELS_KEYPHRASES_CSV_PATH)
    output_df = x.get(4)
    print(output_df.MSWC_ID)
    print(output_df.keyword)
    print(output_df)

    print("Concise TED Results")
    output_rows = x.get_ted_talk_id_and_transcript(4)
    print(output_rows)

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
