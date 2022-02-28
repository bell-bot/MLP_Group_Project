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


import pandas as pd
from random import randint
import numpy as np
from src.datasets import TEDLIUMCustom, MultiLingualSpokenWordsEnglish, DATASET_MLCOMMONS_PATH, DATASET_TEDLIUM_PATH, KeywordsCSVHeaders
import link_utils  
CSV_HEADER = [KeywordsCSVHeaders.KEYWORD, KeywordsCSVHeaders.TED_SAMPLE_ID, KeywordsCSVHeaders.TED_DATASET_TYPE, KeywordsCSVHeaders.MSWC_ID]


#TODO See if all edge cases were handled
class KeywordsLink:
    KEYWORDS_LINK_FILENAME = "keywords.csv"
    
    def __init__(self):
        print("Preparing Ted Dataset...")
        self.TEDLIUMCustomDataset = TEDLIUMCustom(root=DATASET_TEDLIUM_PATH,release="release3")
        print("Preparing MSWC Dataset...")
        self.MSWCDataset = MultiLingualSpokenWordsEnglish(root=DATASET_MLCOMMONS_PATH, read_splits_file=True, subset="train")

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
                    #Append to the list of rows
                    ted_sampleid, dataset_tag, mswc_audioid = self.match(sample_number=i, sample=item, word=word)
                    row = [word, ted_sampleid, dataset_tag, mswc_audioid]
                    csv_rows.append(row)
                else:
                    print(f"Sample id {i} contained no word to link to the keyword dataset.")
                    samples_with_no_links.append(i)
            w.writerows(csv_rows)

        
        #Log words that did not make it to the csv file
        self.create_log_files(not_found, error_words, samples_with_no_links)


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


        example_transcript = link_utils.preprocess_text(transcript)
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
                if link_utils.is_number(word):
                    word = link_utils.parse_number_string(word)
                if word in self.MSWCDataset.keywords:
                    break
                else:
                    not_found = link_utils.append_freq(word, sample_num, not_found)


            except Exception as e:
                print("Something went wrong:")
                print(traceback.print_exc())
                print(f"Sample {sample_num} for word {word}: Choosing another keyword for now")
                error_words = link_utils.append_freq(word, sample_num, error_words)

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
            errors_file[talk_id] = "No Dataset Type Tag Error"
        ####### Ted Audio ID
        ted_sampleid = sample_number
        ####### MSWC Audio ID
        #Retrieve random word from the list of all keywords recordings available in the MSWC dataset
        random_number = randint(0,len(labels_df)-1)
        mswc_audioid = labels_df.iloc[random_number]["LINK"]

        return ted_sampleid, dataset_tag, mswc_audioid, errors_file

    #### -------Extra Helper Functions ------- ####

    def create_log_files(self, not_found, error_words, samples_with_no_links):
        link_utils.log_words_to_ids_in_csv(filename="logs/not_found.csv", dict_words_to_ids= not_found)
        link_utils.log_words_to_ids_in_csv(filename="logs/error_parsing.csv", dict_words_to_ids= error_words)
        link_utils.log_ids_in_csv(filename="logs/samples_with_no_links.csv", list_of_ids=samples_with_no_links)





    def find_match(self,i):
        not_found, error_words, errors_file, sample_with_no_link = {} , {}, {}, None
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
            sample_with_no_link = i
        queue.put([i,row, not_found, error_words, errors_file, sample_with_no_link]) 



if __name__== "__main__":
    #Change working directory to where the script is
    os.chdir(os.path.abspath(os.path.dirname(__file__))) 
    #Prepare KeywordsLink class
    linkerEngine = KeywordsLink()
    not_found = defaultdict(list)
    samples_with_no_links =  []
    error_words = defaultdict(list)
    csv_file =  open(linkerEngine.KEYWORDS_LINK_FILENAME, "w")
    not_found_file =  open("logs/not_found.json", "w")
    error_parsing_file =  open("logs/error_parsing.json", "w")
    samples_no_links_file =  open("logs/samples_with_no_links.json", "w")
    errors_file =  open("logs/errors.json", "w")

    w = csv.writer(csv_file)
    # not_found_w = csv.writer(csv_file_2)
    # error_w = csv.writer(csv_file_3)
    # samples_no_links_w = csv.writer(csv_file_4)

    w.writerow(CSV_HEADER)
    # not_found_w.writerow(["word", "TED_id"])
    # error_w.writerow(["word", "TED_id"])
    # samples_no_links_w.writerow(["TED_id"])

    queue = Queue()
    number_of_items = linkerEngine.TEDLIUMCustomDataset.__len__()
    number_of_items = 110
    def consume():
        while True:
            if not queue.empty():
               
                i,row, not_found, error_words, error_types, sample_with_no_link = queue.get()
                # print("QUEUE", i)
                if (i%1000==0):
                    print(f"----- Sample {i} out of {number_of_items}-----")

                #Ensure finals rows are written
                if row !=[]:
                    w.writerow(row)
                
                # for word, ted_id in not_found.items():
                if not_found != {}:
                    json.dump(not_found, not_found_file)

                # for word, ted_id in error_words.items():
                #     error_w.writerow([word,ted_id])
                if error_words != {}:
                    json.dump(error_words, error_parsing_file)
            
                if error_types != {}:
                    json.dump(error_types, errors_file)

                if sample_with_no_link != None:
                    # samples_no_links_w.writerow([sample_with_no_link])
                    json.dump({"sample_number": sample_with_no_link}, samples_no_links_file)


                if i == number_of_items-1:
                    return
    
    consumer = Thread(target=consume)
    consumer.setDaemon(True)
    consumer.start()

       

    with ThreadPoolExecutor(max_workers=20) as executor:
        for i in range(0,number_of_items):
            executor.submit(linkerEngine.find_match, i)



 
    consumer.join()
    csv_file.close()
    not_found_file.close()
    error_parsing_file.close()
    errors_file.close()

    samples_no_links_file.close()

  
##COMMENTS: Fixed multithreading + JSON files for logs instead