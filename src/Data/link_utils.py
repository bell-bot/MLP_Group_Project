import csv 
import regex as re
from num2words import num2words
"""
Utility functions to help in generating keywords from MSWC dataset
"""

#TODO: Write unit tests to test functionality of preprocessing text


###### ------- Text Preprocessing and utilities --------- ######

#Edge case: HTML5
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
    number = int(re.findall(r"\d+", inputString)[0])
    return numbers_to_words[number]

def parse_number_string(word):
    word = word.lower() #Preprocess to make sure we always deal with lower case letters (eg. 11th)
    if not has_number(word):
        return word
    if is_ordinal(word):
        word = re.sub(r'th', "", word)
        word = num2words(word, to="ordinal")
    elif is_abbreviated_decades(word):
        word = get_abbreviated_number_word_form(word)
    elif is_number_only(word):
        word = num2words(word)
    else:
        print(f"Did not match any specific numbering cases: {word}")
    return word



def handle_apostrophes_in_words(regex, string):
    pass

def handles_identifiers():
    list_of_identifiers = ["<unk>"]
    pass

def preprocess_text(string):
    string=  string.strip().lower()
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