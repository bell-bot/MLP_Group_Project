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
from datasets import TEDLIUMCustom, MultiLingualSpokenWordsEnglish, DATASET_MLCOMMONS_PATH, DATASET_TEDLIUM_PATH
CSV_HEADER = ["Keyword", "TEDLIUM_AudioFileID", "TEDLIUM_SET", "MSWC_AudioFileID"]


#TODO! Check which words were not linked to
#TODO MAke a Labeldataset to read from your file, split them up
#TODO See if all edge cases were handled
#TODO! Ensure same sampling rate
class LabelCreation:
    pass

#Edge case: HTML5
def is_word_with_number(inputString):
    return bool(re.search(r'\d+', inputString))


def is_number(inputString):
    return bool(re.search(r'\b\d+\b', inputString))


def is_ordinal(inputString):
    return bool(re.search(r'(\d+th)', inputString)) or bool(re.search(r'(\d+st)', inputString)) or bool(re.search(r'(\d+nd)', inputString)) or bool(re.search(r'(\d+rd)', inputString))

def is_abbreviated_decades(inputString):
    return bool(re.search(r'(\')?\d+\'?s', inputString))


def get_abbreviated_number_word_form(inputString):
    numbers_to_words = {
        10: "tens",
        20: "twenties",
        30: "thirties",
        40: "fourties",
        50: "fifties",
        60: "sixties",
        70: "seventies",
        80: "eighties" ,
        90: "nineties" ,
    }
    number = int(re.findall(r"\d+", inputString)[0])
    return numbers_to_words[number]

def parse_number_string(word):
    if is_ordinal(word):
        word = re.sub(r'th', "", word)
        word = num2words(word, to="ordinal")
    elif is_abbreviated_decades(word):
        word = get_abbreviated_number_word_form(word)
    else:
        word = num2words(word)
    return word



def handle_apostrophes_in_words(regex, string):
    pass

def handles_identifiers():
    list_of_identifiers = ["<unk>"]
    pass

def preprocess_text(string):
    return string.strip().lower()

def log_to_csv(filename, dict_words_to_ids):
    with open(filename) as f:
        f = csv.writer(csv_file)
        f.writerow(["word", "TED_id"])
        for word, audioids in dict_words_to_ids.items():
            for id in audioids:
                f.writerow([word,id])

def append_freq(word, id, dict_word_to_audio_id):
    if dict_word_to_audio_id[word] == None:
        dict_word_to_audio_id[word] = []
    dict_word_to_audio_id[word].append(i)
    return dict_word_to_audio_id

if __name__== "__main__":
    TEDLIUMCustomDataset = TEDLIUMCustom(root=DATASET_TEDLIUM_PATH,release="release3")
    MSWCDataset = MultiLingualSpokenWordsEnglish(root=DATASET_MLCOMMONS_PATH, read_splits_file=True, subset="train")

   
    not_found = defaultdict(list)
    samples_with_no_links =  []
    error_words = defaultdict(list)
    with open("keywords_link.csv", "w") as csv_file:
        w = csv.writer(csv_file)
        w.writerow(CSV_HEADER)
        number_of_items = TEDLIUMCustomDataset.__len__()
        csv_rows = []
        for i in range(0,number_of_items):
            item = TEDLIUMCustomDataset.__getitem__(i)
            transcript = item["transcript"]
            talk_id = item["talk_id"]
            if (i%1000==0):
                print(f"----- Sample {i} -----")
                w.writerows(csv_rows)
                csv_rows = []

            example_transcript = preprocess_text(transcript)
            #Handles edge case in transcripts where a word may have a space before an apostrophe.
            #i.e) "didn' t" to "didn't"
            regex = re.compile(r" (?=(['\"][a-zA-Z0-9_]))")
            string = regex.sub(r"", example_transcript)
            tokens = string.split(" ")
            token_choice = np.random.permutation(len(tokens))
            word = None
            for choice_index in token_choice:
                word = tokens[choice_index]
                print(f"{word} in {i}")

                try:
                    if is_number(word):
                        word = parse_number_string(word)
                    if word in MSWCDataset.keywords:
                        break
                    else:
                        not_found = append_freq(word, i, not_found)


                except Exception as e:
                    print(e)
                    print("Sample {i} for word {word}: Choosing another keyword for now")
                    error_words = append_freq(word, i, error_words)


            if word in MSWCDataset.keywords:
                labels_df = MSWCDataset.splits_df[MSWCDataset.splits_df["WORD"] == word]
                labels_df.reset_index(inplace=True)
                random_number = randint(0,len(labels_df)-1)
                
                keyword = labels_df.iloc[random_number]['WORD']
                ted_audioid = i
                dataset_tag = None
                for dataset_type, talk_ids in TEDLIUMCustomDataset.recordings_set_dict.items():
                    if talk_id in talk_ids:
                        dataset_tag = dataset_type
                        break
                if dataset_tag == None:
                    print("SOMETHING WENT WRONG! transcript: {}, talk_id: {}")
                    assert(False)
                mswc_audioid = labels_df.iloc[random_number]["LINK"]
                row = [keyword, ted_audioid, dataset_tag, mswc_audioid]
                csv_rows.append(row)
            else:
                # print(f"--{word} doesn't exist in the Keyword Dataset.")
                samples_with_no_links.append(i)
        w.writerows(csv_rows)

            

    log_to_csv(filename="not_found.csv", dict_words_to_ids= not_found)
    log_to_csv(filename="error_parsing.csv", dict_words_to_ids= error_words)
    with open("TED_ids_no_keywords") as f:
        f = csv.writer(csv_file)
        f.writerow(["TED_id"])
        for id in samples_with_no_links:
            f.writerow(id)
