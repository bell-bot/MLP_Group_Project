import os 
from tqdm import tqdm
from sphfile import SPHFile

PATH = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/Data/TEDLIUM_release-3/data/sph/'
PATH_train = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/Data/TEDLIUM_release-3/legacy/train/sph/'
PATH_test = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/Data/TEDLIUM_release-3/legacy/test/sph/'
PATH_dev = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/Data/TEDLIUM_release-3/legacy/dev/sph/'
PATH_train_wav = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/Data/TEDLIUM_release-3/legacy/train/wav/'
PATH_dev_wav = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/Data/TEDLIUM_release-3/legacy/dev/wav/'
PATH_test_wav = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/Data/TEDLIUM_release-3/legacy/test/wav/'

new_target_wav_paths_dict = {"train":PATH_train_wav , "dev":PATH_dev_wav , "test": PATH_test_wav }
source_paths = {"train":PATH_train , "dev":PATH_dev , "test": PATH_test }

# TEDLIUM_SAMPLE_RATE = 16000

class Convert2Wav:
    def __init__(self):

        for folder_set_name, path_to_sph in source_paths.items():
            print("--------------------------------")
            folder = os.fsencode(path_to_sph)
            filenames = []
            folderpath = []
            outputfile = []
            for file in os.listdir(folder):
                filename = os.fsdecode(file)
                if filename.endswith(('.sph')): # whatever file types you're using...\n",
                    filenames.append(filename)

            length =  len(filenames)

            print("Working on {}".format(folder_set_name))
            for i in tqdm(range(length),desc="Getting sph files"):
                fpath = os.path.join(path_to_sph+filenames[i])
                folderpath.append(fpath)
                outpath = os.path.join(filenames[i][:-4]+".wav")
                outputfile.append(outpath)



                        
            for i in tqdm(range(length), desc="Writing WAV files"):
                sph =SPHFile(folderpath[i])
                sph.write_wav(os.path.join(new_target_wav_paths_dict[folder_set_name],outputfile[i])) # Customize the period of time to crop


if __name__== "__main__":
    x= Convert2Wav()
