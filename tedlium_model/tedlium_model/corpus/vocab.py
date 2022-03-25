import os

PATH_TO_DATA = "/home/szy/Documents/code/End-to-end-ASR-Pytorch/data/TEDLIUM_release-3"
DATA_SPLIT = ["dev", "test", "train"]
OUTPUT_FILENAME = "word_vocab.txt"
##############Get Unique Words###############################################################################
unique_words = set()
for data_set in DATA_SPLIT:
    word_list = []
    text_file = os.path.join(PATH_TO_DATA, data_set, "text")
    print(text_file)
    with open(text_file) as f:
        lines = f.readlines()
        for line in lines:
            split = line.split(" ")
            for i in range(1, len(split)):
                if "<unk>" not in split[i]:
                    if("\n" in split[i]):
                        split[i] = split[i].strip()
                    word_list.append(split[i])
    to_set = set(word_list)
    unique_words = unique_words.union(to_set)

##############################################################################################################
##############Write Unique words to file######################################################################
with open(OUTPUT_FILENAME, 'w') as f:
    for word in unique_words:
        f.write(word + "\n")
##############################################################################################################