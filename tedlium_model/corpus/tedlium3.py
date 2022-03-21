import csv
from tkinter.messagebox import NO
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset

# Additional (official) text src provided
OFFICIAL_TXT_SRC = ['TEDLIUM_release-3.tgz']
# Remove longest N sentence in librispeech-lmlibrispeech.py-norm.txt
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading LibriSpeech
READ_FILE_THREADS = 4


def read_text(file):
    '''Get transcription of target wave file, 
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    #print("file")
    #print(file)
    #print(file.split('sph')[0])

    #print("srcandidx")
    src_file = file.split('sph')[0] +'text'
    #print(src_file)
    idx = file.split('/')[-1].split('.')[0]
    idx2 = file.split('/')[-1].split('.')[1]
    #print(idx)
    with open(src_file, 'r') as fp:
        # print("LINES")
        for line in fp:

            #print(line)
            #print("COMPARE")
            # print("FILENAME: {}".format(line.split('-')[0]))
            # print(idx)
            # print(idx2)
            #print(line[:-1].split(' ', 1)[1])
            if idx == line.split(' ')[0].split("-")[0]:
                # print("FOUND")
                return line[:-1].split(' ', 1)[1]


class Ted3Dataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        # List all wave files
        # print(self.path)
        file_list = []
        print(split)
        for s in split:
            split_list = list(Path(join(path, s)).rglob("sph/*.sph"))
            print(len(split_list))
            assert len(split_list) > 0, "No data found @ {}".format(join(path,s))
            file_list += split_list
        # Read text
        # print(file_list)
        file_test = read_text(str(file_list[0]))
        # print("FILE_TEST")
        # print(file_test)

        list_of_texts = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        #text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        # print("text")

        # print(text)
        text = []
        encoding_file_name = None
        if split[0] == "train":
            encoding_file_name = "encoding_errors_train.csv"
        if split[0] == "dev":
            encoding_file_name = "encoding_errors_dev.csv"
        if split[0] == "test":
            encoding_file_name = "encoding_errors_test.csv"
        with open(encoding_file_name, "w") as encoding_f:
            csv_w = csv.writer(encoding_f)
            csv_w.writerow(["count", "transcript"])

            count = 0
            for txt in list_of_texts:
                count+=1
                try:
                    text.append(tokenizer.encode(txt))
                except Exception as e:
                    print("SOMETHING WENT WRONG ENCODING THIS TRANSCRIPT: {}".format(txt))
                    print(e)
                    csv_w.writerow([file_list[count], txt])


        # Sort dataset by text length
        #file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)


class Ted3TextDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.encode_on_fly = False
        read_txt_src = []

        # List all wave files
        file_list, all_sent = [], []

        for s in split:
            if s in OFFICIAL_TXT_SRC:
                self.encode_on_fly = True
                with open(join(path, s), 'r') as f:
                    all_sent += f.readlines()
            file_list += list(Path(join(path, s)).rglob("*.flac"))
        assert (len(file_list) > 0) or (len(all_sent)
                                        > 0), "No data found @ {}".format(path)

        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        all_sent.extend(text)
        del text

        # Encode text
        if self.encode_on_fly:
            self.tokenizer = tokenizer
            self.text = all_sent
        else:
            self.text = [tokenizer.encode(txt) for txt in tqdm(all_sent)]
        del all_sent

        # Read file size and sort dataset by file size (Note: feature len. may be different)
        self.text = sorted(self.text, reverse=True, key=lambda x: len(x))
        if self.encode_on_fly:
            del self.text[:REMOVE_TOP_N_TXT]

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text)-self.bucket_size, index)
            if self.encode_on_fly:
                for i in range(index, index+self.bucket_size):
                    if type(self.text[i]) is str:
                        self.text[i] = self.tokenizer.encode(self.text[i])
            # Return a bucket
            return self.text[index:index+self.bucket_size]
        else:
            if self.encode_on_fly and type(self.text[index]) is str:
                self.text[index] = self.tokenizer.encode(self.text[index])
            return self.text[index]

    def __len__(self):
        return len(self.text)
