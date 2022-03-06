import csv 
import regex as re
from num2words import num2words
"""
Utility functions to help in generating keywords from MSWC dataset
"""

#TODO: Write unit tests to test functionality of preprocessing text


###### ------- Text Preprocessing and utilities --------- ######

#Edge case: HTML5
def split_num_from_word(inputString):
    return re.sub(r"([0-9]+(\.[0-9]+)?)", r" \1 ", inputString).strip()


def has_number(inputString):
    return bool(re.search(r'\d+', inputString))


# #TODO: Fix this,
def is_number_only(inputString):
    return bool(re.search(r'\b\d+\b', inputString))

#Example: 11th , 23rd, etc.
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
    
    inputString = re.sub("s","", inputString).strip()
    if len(inputString) != 4:
        return numbers_to_words[int(inputString)]
    else:
        first_num_string = inputString[0:2]
        second_num_string = inputString[2:4]
        word_form = num2words(first_num_string) + " " + numbers_to_words[second_num_string]
        return word_form

def parse_number_string(word):
    word = word.lower() #Preprocess to make sure we always deal with lower case letters (eg. 11th)
    if not has_number(word):
        return word
    is_parsed_flag = True
    if is_ordinal(word):
        word = re.sub(r'th', "", word)
        word = re.sub(r'st', "", word)
        word = re.sub(r'nd', "", word)
        word = re.sub(r'rd', "", word)
        word = num2words(word, to="ordinal")
    elif is_abbreviated_decades(word):
        word = get_abbreviated_number_word_form(word)
    elif is_number_only(word):
        word = num2words(word)
        word = word.replace("-", " ")
    else:
        try:
            word  = split_num_from_word(word)
            tokens = [num2words(token) if is_number_only(token) else token for token in word.split()]
            word = ' '.join(tokens)
        except:
            is_parsed =  False
            print(f"Did not match any specific numbering cases: {word}")
        
    if is_parsed_flag:
        word = word.replace("-", " ") #Example: 25th : twenty-fifth becomes twenty fifth

    return word



def handle_apostrophes_in_words(regex, string):
    pass

def handle_acronyms_edge_cases():
    pass
    #Example: at & t's -> AT&T's

#NOTE: Edge case to deal with pronouncing certain symbols. However, it introduces errors (= can be pronounced as equals or equal). 
#TODO: See if there is a library that handles, check nltk
def handle_pronouncing_symbols(string):
    string=  string.replace("="," equal ")
    # string=  string.replace("-"," minus ")
    # string=  string.replace("<"," less than ")
    # string=  string.replace(">"," greater than ")
    string=  string.replace("$"," dollars ")
    string=  string.replace("&"," and ")

    return string


def handles_identifiers():
    list_of_identifiers = ["<unk>"]
    pass

def preprocess_text(string):
    string=  string.strip().lower()
    string = handle_pronouncing_symbols(string)
    #remove white (duplicate) space
    regex = re.compile(r'\s+')
    string = ' '.join(re.split(regex, string))
    #Handles edge case in transcripts where a word may have a space before an apostrophe.
    #i.e) "didn' t" to "didn't"
    regex = re.compile(r" (?=(['\"][a-zA-Z0-9_]))")
    string = regex.sub(r"", string)
    #Remove apostrophe if it exists at the end of a word 
    # i.e) others' to others, shapes' to shapes, etc.)
    regex = re.compile(r"\'(?=[^\w])")
    string = regex.sub(r"", string)


    return string


##### ------------- Logging -------------- ######

def log_words_to_ids_in_csv(filename, dict_words_to_ids):
    with open(filename, "w") as csv_file:
        f = csv.writer(csv_file)
        f.writerow(["word", "TED_id"])
        for word, audioids in dict_words_to_ids.items():
            for id in audioids:
                f.writerow([word,id])

def log_ids_in_csv(filename,list_of_ids):
    with open(filename, "w") as csv_file:
        f = csv.writer(csv_file)
        f.writerow(["TED_id"])
        for id in list_of_ids:
            f.writerow(id)

#### ------------ Extra ------------ #########

def append_freq(word, id, dict_word_to_audio_id):
    if dict_word_to_audio_id.get(word) == None:
        dict_word_to_audio_id[word] = []
    dict_word_to_audio_id[word].append(id)
    return dict_word_to_audio_id