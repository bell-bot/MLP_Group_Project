
import os
from typing import Dict, Tuple
import sys
import logging
import regex as re
import torchaudio.datasets.tedlium as tedilum 
import torchaudio
from torch import Tensor
import pandas as pd


"""Specify path to TEDLIUM directory"""
DATASET_TEDLIUM_PATH = "../../../Data/"
DATASET_MLCOMMONS_PATH = "../../../Data/"
LABELS_PATH = "./labels.csv"

class KeywordsCSVHeaders:
    KEYWORD = "Keyword"
    TED_ID= "TEDLIUM_AudioFileID"
    TED_DATASET_TYPE = "TEDLIUM_SET"
    MSWC_ID = "MSWC_AudioFileID"


#TODO! Customise for each subset, in speaker-adaptation. Might require changing the metadata
class TEDLIUMCustom(tedilum.TEDLIUM):
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
    def __init__(self, root, release= "release3", subset=None):
        super().__init__(root, release=release)
     
        path_to_speaker_adaptation = os.path.join(root, tedilum._RELEASE_CONFIGS[release]["folder_in_archive"], "speaker-adaptation")
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

    def _load_audio(self, path: str, start_time: float, end_time: float, sample_rate: int = 16000) -> [Tensor, int]:
        return super()._load_audio(path, start_time, end_time, sample_rate)

    def __getitem__(self, AudioFileID: int) -> Tuple[Tensor, int, str, int, int, int, str, str]:

        """Load the n-th sample from the dataset, where n is the audioFileID

        Args:
            AudioFileID (int): The index of the sample to be loaded, which is also termed as the unique ID

        Returns:
            tuple: ``(waveform, sample_rate, transcript, talk_id, speaker_id, identifier, start_time, end_time)`` 
            """
        return super().__getitem__(AudioFileID)

    def _get_specific_audio(self, audio_id):
        pass

    def _load_tedlium_item(self, fileid: str, line: int, path: str) -> Dict:
        """Loads a TEDLIUM dataset sample given a file name and corresponding sentence name. Functionality taken from original source code.
        
        ----> Custom function returns start time and end time as well 

        Args:
            fileid (str): File id to identify both text and audio files corresponding to the sample
            line (int): Line identifier for the sample inside the text file
            path (str): Dataset root path

        Returns:
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
            "start_time": start_time,
            "end_time": end_time, 
        }
        return results_dict




class MultiLingualSpokenWordsEnglish():
    MLCOMMONS_FOLDER_NAME = "Multilingual_Spoken_Words"
    AUDIO_DIR_NAME="en"
    SPLITS_DIR_NAME="splits"
    ALIGNMENTS_DIR_NAME="alignments"

 

    def raise_directory_error():
        raise RuntimeError(
            f"Please configure the path to the Spoken Keywords Dataset, with the directory name \"{self.MLCOMMONS_FOLDER_NAME}\", containing the three subfolders:" \
            + "\n" + \
            f"\"{self.AUDIO_DIR_NAME}\" for audio, \"{self.SPLITS_DIR_NAME}\" for splits directory, and \"{self.ALIGNMENTS_DIR_NAME}\" for alignemnts directory"
        )

    #TODO! Accept 4 kinds of values: Train vs test vs Dev vs "all"
    def __init__(self, root, read_splits_file=False, subset="all") -> None:
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



        
        #Retrieve the splits csv file
        if read_splits_file:
            self._path_to_splits = os.path.join(self._path, self._subfolder_names_dict["splits"])
            self.splits_df = pd.read_csv(os.path.join(self._path_to_splits, "en_splits.csv"))
            if subset == "train":
                self.splits_df = self.splits_df[self.splits_df["SET"] == "TRAIN"]
            #Extra step to preprocesses words to one form of apostrophe
            self.splits_df["WORD"].replace("`|’", "'", regex=True, inplace=True) 
            #Retrieve the words that have been validated as True, affirming that the spoken audio matches the transcription 
            self.splits_df = self.splits_df[self.splits_df["VALID"] == True]
            #Retrieve the keywords in the dataset
            self.keywords = set(self.splits_df["WORD"].unique())


    def _load_audio(self, path_to_audio) -> [Tensor, int]:
        """Loads audio data from file given file path
        
        Returns:
            waveform: Tensor
            sample_rate: int
        """
        return torchaudio.load(path_to_audio)


    def __getitem__(self, MSWC_AudioID) -> Dict:
        """Retrieves sample data from file given Audio ID
        """
        path_to_audio = os.path.join(self._path, "en", "clips", MSWC_AudioID)
        waveform, sample_rate= self._load_audio(path_to_audio)
        results_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate 
        }
        return results_dict


class CTRLF_LinkedDataset:
    def __init__(self,path_to_keywords_csv, path_to_TED=DATASET_TEDLIUM_PATH, path_to_MSWC=DATASET_MLCOMMONS_PATH, single_words=True):
        self._path_to_TED = path_to_TED
        self._path_to_MSWC = path_to_MSWC
        #Initialise keyword dataframe
        keywords_df = pd.read_csv(path_to_keywords_csv)
        self.audio_keywords_dataset_dict = {
            "train": keywords_df[keywords_df["TEDLIUM_SET"] == "train"],
            "dev": keywords_df[keywords_df["TEDLIUM_SET"] == "dev"],
            "test": keywords_df[keywords_df["TEDLIUM_SET"] == "test"]
        }

        #Initialise Ted talk dataset
        self.TED = TEDLIUMCustom(root=path_to_TED,release="release3")

        #Initialise Keyword dataset
        self.MSWC = MultiLingualSpokenWordsEnglish(root=path_to_MSWC)


 

    def get_item(self, TEDAudio_id: int, dataset_type: str):
        subset_keywords_df = self.audio_keywords_dataset_dict[dataset_type]

        MSWC_audio_ids = subset_keywords_df[subset_keywords_df[KeywordsCSVHeaders.TED_ID] == TEDAudio_id]
        TED_results_dict = self.TED.__getitem__(TEDAudio_id)
        MSWC_results_dict =  self.MSWC.__getitem__(MSWC_audio_ids[KeywordsCSVHeaders.MSWC_ID][0])

        return TED_results_dict, MSWC_results_dict

if __name__== "__main__":

    x= CTRLF_LinkedDataset("./metadata.csv")
    Ted_dict, MSWC_dict= x.get_item(0, dataset_type="train")
    print(Ted_dict, MSWC_dict)
