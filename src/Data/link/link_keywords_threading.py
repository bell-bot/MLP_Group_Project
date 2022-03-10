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
from src.datasets import TEDLIUMCustom, MultiLingualSpokenWordsEnglish, DATASET_MLCOMMONS_PATH, DATASET_TEDLIUM_PATH, KeywordsCSVHeaders, KEYWORDS_LINK_CSV_PATH
import src.Data.data_utils  as data_utils
CSV_HEADER = [KeywordsCSVHeaders.KEYWORD, KeywordsCSVHeaders.TED_SAMPLE_ID, KeywordsCSVHeaders.TED_DATASET_TYPE, KeywordsCSVHeaders.MSWC_ID]


#TODO See if all edge cases were handled
class KeywordsLink:
    KEYWORDS_LINK_FILENAME = KEYWORDS_LINK_CSV_PATH
    PATH_TO_LOGS = "../logs"
    NOT_FOUND_LOG = os.path.join(PATH_TO_LOGS, "not_found.csv")
    ERROR_PARSING_LOG = os.path.join(PATH_TO_LOGS, "error_parsing.csv")
    SAMPLE_NO_SUITABLE_LINKS_LOG = os.path.join(PATH_TO_LOGS, "samples_with_no_links.csv")
    EXECUTION_ERROR_LOG = os.path.join(PATH_TO_LOGS, "errors.csv")

    def __init__(self, overwrite=False):
        print("Preparing Ted Dataset...")
        self.TEDLIUMCustomDataset = TEDLIUMCustom(root=DATASET_TEDLIUM_PATH,release="release3")
        print("Preparing MSWC Dataset...")
        self.MSWCDataset = MultiLingualSpokenWordsEnglish(root=DATASET_MLCOMMONS_PATH, read_splits_file=True, subset="train")

        if overwrite:
            self.access_mode = "w" #Access mode set to create a new file (overwrites file if it exists)
            self.last_read_sample_id = 0
        else:
            self.access_mode = "a" #Access mode to append to csv
            self.last_read_sample_id = self.retrieve_last_sample_id()

    #Reads from CSV file the last read sample id, to continue from last time we stopped
    #TODO: Multithreading mixes order, so might need to start from a more specific spot
    #TODO: add error handling similar to alignments.py
    def retrieve_last_sample_id(self):
        ted_sample_id_column = 1
        #NOTE: Assert that the order is the same, though not a robust solution, it was done. The additional assertion check is done to make sure we are not reading from another column
        assert(KeywordsCSVHeaders.TED_SAMPLE_ID == CSV_HEADER[ted_sample_id_column])
        #subprocess requires byte like object to read
        line = subprocess.check_output(['tail', '-1', bytes(self.KEYWORDS_LINK_FILENAME, encoding="utf-8")])
        print(line)
        #Convert from bytes to int 
        last_read_sample_id = int(str(line.split(b',')[ted_sample_id_column], encoding="utf-8"))
        last_read_sample_id = last_read_sample_id + 1 #We will start iterating from the sample id after the last one
        print(f"Side note: The python script assumes that the TED Sample ID column is in {ted_sample_id_column}")
        print(f"Last TED Sample ID Read: {last_read_sample_id}")
        return last_read_sample_id

    ### ----------- Linking Datasets Functions ------------ ###


    ## ---- Keyphrases ---- ##
    def create_keyphrases_csv(self):
        pass


    ## ---- Keywords ---- ##
   

    def create_keywords_csv(self):
        not_found = defaultdict(list)
        samples_with_no_links =  []
        error_words = defaultdict(list)

   
        with open(self.KEYWORDS_LINK_FILENAME, self.access_mode) as csv_file,\
             open(self.NOT_FOUND_LOG, self.access_mode) as not_found_file,\
             open(self.ERROR_PARSING_LOG, self.access_mode) as error_parsing_file,\
             open(self.SAMPLE_NO_SUITABLE_LINKS_LOG, self.access_mode) as samples_no_links_file,\
             open(self.EXECUTION_ERROR_LOG, self.access_mode) as errors_file:

            w = csv.writer(csv_file)
            not_found_w = csv.writer(not_found_file)
            error_parsing_w = csv.writer(error_parsing_file)
            samples_no_links_w = csv.writer(samples_no_links_file)
            errors_w = csv.writer(errors_file)

            if self.access_mode == "w":
                w.writerow(CSV_HEADER)
                not_found_w.writerow(["word", "TED_id"])
                error_parsing_w.writerow(["word", "TED_id"])
                samples_no_links_w.writerow(["TED_id", "talk_id"])
                errors_w.writerow(["TED_talk_id", "Error_type"])

            self.queue = Queue()
            number_of_items = self.TEDLIUMCustomDataset.__len__()
            def consume():
                while True:
                    if not self.queue.empty():
                    
                        i,row, not_found, error_words, error_types, sample_with_no_link = self.queue.get()
                        # print("QUEUE", i)
                        if (i%1000==0):
                            print(f"----- Sample {i} out of {number_of_items}-----")

                        #Ensure finals rows are written
                        if row !=[]:
                            w.writerow(row)
                        
                        # for word, ted_id in not_found.items():
                        if not_found != {}:
                            # json.dump(not_found, not_found_file)
                            for word,ted_id in not_found.items():
                                for i in ted_id:
                                    not_found_w.writerow([word,i])
                        
                        if error_words != {}:
                            for word,ted_id in error_words.items():
                                for i in ted_id:
                                    error_parsing_w.writerow([word,i])

                        if error_types != {}:
                            for talk_id,type_error in error_words.items():
                                errors_w.writerow([talk_id,type_error])

                        if sample_with_no_link != {}:
                            for sample, talk_id in sample_with_no_link.items():
                                samples_no_links_w.writerow([sample,talk_id ])


                        if i == number_of_items-1:
                            return
            
            consumer = Thread(target=consume)
            consumer.setDaemon(True)
            consumer.start()

            

            with ThreadPoolExecutor(max_workers=1) as executor:
                for i in range(self.last_read_sample_id,number_of_items):
                    executor.submit(self.find_match, i)



        
            consumer.join()


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


        string= data_utils.preprocess_text(transcript)
        tokens = string.split(" ")
        token_choice = np.random.permutation(len(tokens))
        word = None
        for choice_index in token_choice:
            word = tokens[choice_index]

            try:
                if data_utils.has_number(word):
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

    #### -------Extra Helper Functions ------- ####

    def create_log_files(self, not_found, error_words, samples_with_no_links):
        data_utils.log_words_to_ids_in_csv(filename="logs/not_found.csv", dict_words_to_ids= not_found)
        data_utils.log_words_to_ids_in_csv(filename="logs/error_parsing.csv", dict_words_to_ids= error_words)
        data_utils.log_ids_in_csv(filename="logs/samples_with_no_links.csv", list_of_ids=samples_with_no_links)





    def find_match(self,i):
        try:
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
            self.queue.put([i,row, not_found, error_words, errors_file, sample_with_no_link]) 
        except:
            print(traceback.print_exc())
            sys.exit(-1)



if __name__== "__main__":
    #Change working directory to where the script is
    os.chdir(os.path.abspath(os.path.dirname(__file__))) 
    #Prepare KeywordsLink class
    linkerEngine = KeywordsLink(overwrite=False)
    linkerEngine.create_keywords_csv()

  
