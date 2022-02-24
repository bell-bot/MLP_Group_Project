from collections import defaultdict
import os
from typing import Tuple
import sys
import logging
from numpy import number
import regex as re
from num2words import num2words
import csv

from torch import Tensor
from torch.utils.data import Dataset


import pandas as pd
from random import randint
import numpy as np
from src.datasets import TEDLIUMCustom, MultiLingualSpokenWordsEnglish, DATASET_MLCOMMONS_PATH, DATASET_TEDLIUM_PATH, KeywordsCSVHeaders
import link_utils  
CSV_HEADER = [KeywordsCSVHeaders.KEYWORD, KeywordsCSVHeaders.TED_ID, KeywordsCSVHeaders.TED_DATASET_TYPE, KeywordsCSVHeaders.MSWC_ID]


#TODO See if all edge cases were handled
class KeywordsLink:
    KEYWORDS_LINK_FILENAME = "keywords.csv"
    
    def __init__(self):
        self.TEDLIUMCustomDataset = TEDLIUMCustom(root=DATASET_TEDLIUM_PATH,release="release3")
        print("Preparing Ted Dataset...")
        self.MSWCDataset = MultiLingualSpokenWordsEnglish(root=DATASET_MLCOMMONS_PATH, read_splits_file=True, subset="train")
        print("Preparing MSWC Dataset...")

    ### ----------- Linking Datasets Functions ------------ ###


    ## ---- Keyphrases ---- ##
    def create_keyphrases_csv(self):
        pass


    ## ---- Keywords ---- ##

    def create_keywords_csv(self):
        #Initialise dictionaries and list which store log information 
        not_found = defaultdict(list)
        samples_with_no_links =  []
        error_words = defaultdict(list)
        with open(self.KEYWORDS_LINK_FILENAME, "w") as csv_file:
            w = csv.writer(csv_file)
            w.writerow(CSV_HEADER)
            number_of_items = self.TEDLIUMCustomDataset.__len__()
            csv_rows = []
            for i in range(0,number_of_items):
                if (i%1000==0):
                    print(f"----- Sample {i} -----")
                    w.writerows(csv_rows)
                    csv_rows = []

                item = self.TEDLIUMCustomDataset.__getitem__(i)
                word, not_found, error_words = self.get_keyword_from_TED_audio_sample(sample_num=i, item_sample=item, not_found=not_found, error_words= error_words)
                if word in self.MSWCDataset.keywords:
                    keyword, ted_audioid, dataset_tag, mswc_audioid = self.match(sample_number=i, sample=item, word=word)
                    row = [keyword, ted_audioid, dataset_tag, mswc_audioid]
                    csv_rows.append(row)
                else:
                    print(f"Sample id {i} contained no word to link to the keyword dataset.")
                    samples_with_no_links.append(i)
            w.writerows(csv_rows)

        
        #Log words that did not make it to the csv file
        self.create_log_files(not_found, error_words, samples_with_no_links)

    def get_keyword_from_TED_audio_sample(self, sample_num, item_sample, not_found, error_words):
    
        transcript = item_sample["transcript"]


        example_transcript = link_utils.preprocess_text(transcript)
        #Handles edge case in transcripts where a word may have a space before an apostrophe.
        #i.e) "didn' t" to "didn't"
        regex = re.compile(r" (?=(['\"][a-zA-Z0-9_]))")
        string = regex.sub(r"", example_transcript)
        tokens = string.split(" ")
        token_choice = np.random.permutation(len(tokens))
        word = None
        for choice_index in token_choice:
            word = tokens[choice_index]
            print(f"{word} in {sample_num}")

            try:
                if link_utils.is_number(word):
                    word = link_utils.parse_number_string(word)
                if word in self.MSWCDataset.keywords:
                    break
                else:
                    not_found = link_utils.append_freq(word, sample_num, not_found)


            except Exception as e:
                print("Something went wrong:")
                print(e)
                print(f"Sample {sample_num} for word {word}: Choosing another keyword for now")
                error_words = link_utils.append_freq(word, sample_num, error_words)

        return word, not_found, error_words

    def match(self,sample_number, sample, word):
        labels_df = self.MSWCDataset.splits_df[self.MSWCDataset.splits_df["WORD"] == word]
        labels_df.reset_index(inplace=True)

        #Retrieve random word from the list of all keywords recordings available in the MSWC dataset
        random_number = randint(0,len(labels_df)-1)
        keyword = labels_df.iloc[random_number]['WORD']
        ted_audioid = sample_number
        dataset_tag = None
        talk_id = sample["talk_id"]
        for dataset_type, talk_ids in self.TEDLIUMCustomDataset.recordings_set_dict.items():
            if talk_id in talk_ids:
                dataset_tag = dataset_type
                break
        if dataset_tag == None:
            print("SOMETHING WENT WRONG! transcript: {}, talk_id: {}")
            assert(False)
        mswc_audioid = labels_df.iloc[random_number]["LINK"]

        return keyword, ted_audioid, dataset_tag, mswc_audioid

    #### -------Extra Helper Functions ------- ####

    def create_log_files(self, not_found, error_words, samples_with_no_links):
        link_utils.log_words_to_ids_in_csv(filename="logs/not_found.csv", dict_words_to_ids= not_found)
        link_utils.log_words_to_ids_in_csv(filename="logs/error_parsing.csv", dict_words_to_ids= error_words)
        link_utils.log_ids_in_csv(filename="logs/samples_with_no_links.csv", list_of_ids=samples_with_no_links)




if __name__== "__main__":
    #Change working directory to where the script is
    os.chdir(os.path.abspath(os.path.dirname(__file__))) 
    #Prepare KeywordsLink class
    linkerEngine = KeywordsLink()

    #Generate keywords csv
    linkerEngine.create_keywords_csv()

  
