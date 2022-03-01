import csv 
import regex as re
"""
Utility functions to help in generating keywords from MSWC dataset
"""


###### ------- Text Preprocessing and utilities --------- ######

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
    if dict_word_to_audio_id[word] == None:
        dict_word_to_audio_id[word] = []
    dict_word_to_audio_id[word].append(id)
    return dict_word_to_audio_id