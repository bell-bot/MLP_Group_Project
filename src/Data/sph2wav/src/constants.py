
import os
from utils import get_git_root

######### ------------------ PATHING ------------- ############
"""Specify path to TEDLIUM directory"""
data_paths = os.path.join(get_git_root(os.getcwd()) ,'Data')
DATASET_TEDLIUM_PATH = data_paths
DATASET_MLCOMMONS_PATH = data_paths
KEYWORDS_LINK_CSV_PATH = os.path.join(data_paths, "KeywordPerSample", "keywords.csv")
KEYPHRASES_LINK_CSV_PATH = os.path.join(data_paths, "Keyphrases" , "keyphrases.csv")


LABELS_KEYPHRASES_CSV_PATH = os.path.join(data_paths, "Keyphrases" , "labels.csv")


############# ---------CSV HEADERS --------------################

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