#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchaudio.datasets import tedlium


# In[3]:


#root_dir = "/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db"
#our_release = "release3"
#split = "test"
#dataset = tedlium.TEDLIUM(root = root_dir,release=our_release,subset=split)

#dataset.__getitem__(1)
#print(dataset.__getitem__(1))


# In[14]:


####Need to convert to wav
import numpy as np
import pandas as pd
import glob
import csv
import librosa


# In[7]:

import numpy as np
import csv
import string

#
# vocabulary table
#

# index to byte mapping
index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# byte to index mapping
byte2index = {}
for i, ch in enumerate(index2byte):
    byte2index[ch] = i

# vocabulary size
voca_size = len(index2byte)


# convert sentence to index list
def str2index(str_):

    # clean white space
    str_ = ' '.join(str_.split())
    # remove punctuation and make lower case
    str_ = str_.translate(string.punctuation).lower()

    res = []
    for ch in str_:
        try:
            res.append(byte2index[ch])
        except KeyError:
            # drop OOV
            pass
    return res


# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for ch in index_list:
        if ch > 0:
            str_ += index2byte[ch]
        elif ch == 0:  # <EOS>
            break
    return str_


# print list of index list
def print_index(indices):
    for index_list in indices:
        print(index2str(index_list))


# real-time wave to mfcc conversion function
def _load_mfcc(src_list):

    # label, wave_file
    label, mfcc_file = src_list

    # decode string to integer
    label = np.fromstring(label, np.int)

    # load mfcc
    mfcc = np.load(mfcc_file, allow_pickle=False)

    # speed perturbation augmenting
    mfcc = _augment_speech(mfcc)

    return label, mfcc


def _augment_speech(mfcc):

    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc = np.roll(mfcc, r, axis=0)

    # zero padding
    if r > 0:
        mfcc[:r, :] = 0
    elif r < 0:
        mfcc[r:, :] = 0

    return mfcc




import os 
from sphfile import SPHFile
path = '/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/data/sph/'
path_train = '/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/legacy/train/sph/'
path_test = '/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/legacy/test/sph/'
path_dev = '/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/legacy/dev/sph/'
path_train_wav = '/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/legacy/train/sph_wav/'


folder = os.fsencode(path_dev)
filenames = []
folderpath = []
outputfile = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith(('.sph')): # whatever file types you're using...\n",
        filenames.append(filename)


length = len(filenames)

#for i in range(length):
#    fpath = os.path.join(path_train+filenames[i])
#    folderpath.append(fpath)
#    outpath = os.path.join("wav_train/" + filenames[i] + ".wav")
#    outputfile.append(outpath)
#    print(outpath)
             
#for i in range(length):
#    sph =SPHFile(folderpath[i])
#    print(sph.format)
#    sph.write_wav(outputfile[i], 0, 123.57 ) # Customize the period of time to crop\n",


# In[35]:


_data_path = "/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/"


# In[41]:


def process_ted(csv_file, category):

    parent_path = _data_path + 'TEDLIUM_release-3/legacy/' + category + '/'
    labels, wave_files, offsets, durs = [], [], [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')
    print(parent_path)
    # read STM file list
    stm_list = glob.glob(parent_path + 'stm/*')
    print(stm_list)
    for stm in stm_list:
        with open(stm, 'rt') as f:
            records = f.readlines()
            for record in records:
                field = record.split()

                # wave file name
                #wave_file = parent_path + 'sph/%s.sph.wav' % field[0]
                wave_file = parent_path + 'sph/%s.sph' % field[0]
                wave_files.append(wave_file)

                # label index
                labels.append(str2index(' '.join(field[6:])))

                # start, end info
                start, end = float(field[3]), float(field[4])
                offsets.append(start)
                durs.append(end - start)

    # save results
    for i, (wave_file, label, offset, dur) in enumerate(zip(wave_files, labels, offsets, durs)):

        # print info
        print("TEDLIUM corpus preprocessing (%d / %d) - '%s-%.2f]" % (i, len(wave_files), wave_file, offset))

        # load wave file
        wave, sr = librosa.load(wave_file, mono=True, sr=None, offset=offset, duration=dur)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # filename
            fn = "%s-%.2f" % (wave_file.split('/')[-1], offset)

            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save('/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/data/preprocessing/mfcc/' + fn + '.npy', mfcc, allow_pickle=False)


# In[ ]:


# TEDLIUM corpus for train
dir_ ="/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/data/preprocessing/meta/train.csv"
csv_f = open(dir_, 'a+')
process_ted(csv_f, 'train')
csv_f.close()


# In[ ]:


# TEDLIUM corpus for valid
dir_ ="/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/data/preprocessing/meta/valid.csv"
csv_f = open(dir_, 'a+')
process_ted(csv_f, 'dev')
csv_f.close()


# In[ ]:


# TEDLIUM corpus for test
dir_ ="/home/szy/Documents/code/espnet/egs/tedlium3/asr1/db/TEDLIUM_release-3/data/preprocessing/meta/test.csv"
csv_f = open(dir_, 'a+')
process_ted(csv_f, 'test')
csv_f.close()


# In[32]:



# In[ ]:




