from collections import defaultdict
import os
from typing import Tuple
import sys
import logging
from numpy import number
import regex as re
from num2words import num2words
import csv
import json

from torch import Tensor
from torch.utils.data import Dataset

from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import traceback
import subprocess

import pandas as pd
from random import randint
import numpy as np
from src.datasets import TEDLIUMCustom, MultiLingualSpokenWordsEnglish, DATASET_MLCOMMONS_PATH, DATASET_TEDLIUM_PATH, KeywordsCSVHeaders
import src.Data.data_utils  as data_utils

class KeywordLinker:

    def __init__(self, TED_Dataset_Object:TEDLIUMCustom , MSWC_Dataset_Object:MultiLingualSpokenWordsEnglish ):
        self.TEDLIUMCustomDataset = TED_Dataset_Object
        self.MSWCDataset = MSWC_Dataset_Object

    ## ---- Keyphrases ---- ##
        def create_keyphrases_csv(self):
            pass
    
    ## ---- Keywords ---- ##
    def link_datasets(self, i: int):
        not_found, error_words, errors_file, sample_with_no_link = {} , {}, {}, {}
        item = self.TEDLIUMCustomDataset.__getitem__(i)
        word, not_found, error_words  = self.get_keyword_from_TED_audio_sample(i, item, not_found, error_words)
        row = []
        if word in self.MSWCDataset.keywords:
            ted_sampleid, dataset_tag, mswc_audioid, errors_file= self.match(i, item, word)
            row = [word, ted_sampleid, dataset_tag, mswc_audioid]
        else:
            print(f"--- Sample id {i} contained no word to link to the keyword dataset.")
            transcript = item["transcript"]
            
            print(f"Transcript: \"{transcript}\" ")
            talk_id = item["talk_id"]

            sample_with_no_link[i] = talk_id
        return i, row, not_found, error_words, errors_file, sample_with_no_link
        # self.queue.put([i,row, not_found, error_words, errors_file, sample_with_no_link]) 

    def match(self,sample_number, sample, word):
        """
        Helper Function that returns relevant information to link the two datasets
        Args:
            sample_number: The sample number of the audio file, also used as the TED_AudioFileId for our purposes 
            sample: The sample of the audio file, containing access to metadata like timestamp and the transcript of that sample
            word: keyword to link the two datasets with
        Returns:
            ted_sampleid: Audio Sample id of the TEDTalk sample. Corresponds to the sample number when calling the TedliumCustom dataset (it is NOT identical to talk_id).
            dataset_tag: The type of dataset that the TED Audio belongs to
            mswc_audioid: Audio id of the keyword from the MSWC dataset

        """
        errors_file = {}
        labels_df = self.MSWCDataset.splits_df[self.MSWCDataset.splits_df["WORD"] == word]
        labels_df.reset_index(inplace=True)
        ####### Dataset Tag (Train vs dev vs test)
        dataset_tag = None
        talk_id = sample["talk_id"]
        transcript= sample["transcript"]
        for dataset_type, talk_ids in self.TEDLIUMCustomDataset.recordings_set_dict.items():
            if talk_id in talk_ids:
                dataset_tag = dataset_type
                break
        if dataset_tag == None:
            print(f"NO DATASET TAG!\n transcript: {transcript}\n, talk_id: {talk_id}\n, sample_num: {sample_number}")
            errors_file[talk_id] = "No Dataset Tag"
        ####### Ted Audio ID
        ted_sampleid = sample_number
        ####### MSWC Audio ID
        #Retrieve random word from the list of all keywords recordings available in the MSWC dataset
        random_number = randint(0,len(labels_df)-1)
        mswc_audioid = labels_df.iloc[random_number]["LINK"]

        return ted_sampleid, dataset_tag, mswc_audioid, errors_file 


    def get_keyword_from_TED_audio_sample(self, sample_num, item_sample, not_found, error_words):
        """
        Helper Function that returns a random keyword from a TED Talk audio sample
        Args:
            sample_num: The sample number of the audio file, also used as the TED_AudioFileId for our purposes
            item_sample: The sample of the audio file, containing access to metadata like timestamp and the transcript of that sample
            not_found: Dictionary maintained while retrieving words that are not found in the Keyword Dataset
            error_words: Words that recieved an error while linking to the Keyword dataset, mostly due to parsing issues.

        Returns:
            word: a random keyword from the audio sample
            not_found
            error_words
        """
        transcript = item_sample["transcript"]


        example_transcript = data_utils.preprocess_text(transcript)
        #Handles edge case in transcripts where a word may have a space before an apostrophe.
        #i.e) "didn' t" to "didn't"
        # regex = re.compile()
        string = re.sub(r" (?=(['\"][a-zA-Z0-9_]))", "", example_transcript)
        tokens = string.split(" ")
        token_choice = np.random.permutation(len(tokens))
        word = None
        for choice_index in token_choice:
            word = tokens[choice_index]

            try:
                if data_utils.is_number(word):
                    word = data_utils.parse_number_string(word)
                if word in self.MSWCDataset.keywords:
                    break
                else:
                    not_found = data_utils.append_freq(word, sample_num, not_found)


            except Exception as e:
                print("Something went wrong:")
                print(traceback.print_exc())
                print(f"Sample {sample_num} for word {word}: Choosing another keyword for now")
                error_words = data_utils.append_freq(word, sample_num, error_words)

        return word, not_found, error_words






