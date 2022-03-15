# Data Structure:
```
+-- Data/
|   +-- Tedlium_release-3/
|       +-- ...
|   +-- Multilingual_Spoken_Words/
|       +-- audio
|       +-- splits
|       +-- alignments
|   +-- scripts/
|       +-- download_ted.sh
|       +-- download_keywords.sh
|   +-- ...
```

Log files are also made during the process of force aligning and linking keywords between the two datasets for potential analysis and book-keeping.

# Downloading Datasets
Please check the scripts folder for more information

### Installing Git-lfs
Note: Due to the huge size of the csv file, you'll need to install git-lfs. Please download git-lfs here: https://git-lfs.github.com/
When you download git-lfs, run the following:
1. `git lfs install`
2. `git add .gitattributes`
 
# File Descriptions
```
+-- Data/
|   +-- KeywordPerSample/
|       +-- labels.csv
|       +-- keywords.csv
|   +-- Keyphrases/
|       +-- labels.csv
|       +-- keyphrases.csv
|       +-- alignments
|   +-- ...
```
## labels.csv 
Contains rows of keyword labels with the timestamps of when it was mentioned in the TED dataset (done by force alignment) along with reference to a keyword audio recording in the MSWC dataset. 




- `Keyword`: str
  -   The keyword linking the two audio files (sample of a TED audio file and an MSWC recording of that keyword)
- `TEDLIUM_SampleID`: int 
  - Represents the id of the audio segment in TED talk. In other words, it is a unique id that maps to a segment of a TED audio file. 
                    This is NOT the same as "TED_TALK_ID", which represents the id of an entire audio file. 
- `TED_TALK_ID`: str
  - A unique id of a TED audio file. Added in order to make referencing the current audio segment/sample to the original TED audio file more accessible
- `TEDLIUM_SET`: str 
  - Either "train", "dev", or "test"
  - The type of dataset the audio sample/segment (or more specifically the audio file) exists in
  - It can be one of the four values: Train vs Dev vs Test vs None
  - Currently, it is not used as it is not possible to work with all the TED data, but there are samples from train, dev, and test 
  - TODO: Not sure if it's imbalanced, will need to put up stats on the distribution 
- `MSWC_AudioID`: str
  - The audio id of the keyword recording from MSWC chosen to link with a TED audio segment. 
  - It is be used to load the audio from MSWC dataset.  
- `start_time`: float
  - The starting of the interval time of when the current keyword was said in the audio segment.
- `end_time`: float
  - The end of the interval time of when the current keyword was said in the audio segment.
- `confidence`: float
  - The average probabilty score of the forced alignment model on generating the keyword timestamp. Represents the model's frame-wise probability from emission matrix. Emission matrix is frame-wise label probability distribution 
  - https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
 
 
 


## keywords.csv/ keyphrases.csv
This is used as an intermediate step to create labels.csv. These contain the same headers as labels.csv, but before force alignment is applied to them (to generate timestamps).

- Side notes:
   keyphrases.csv is very similar to keywords.csv with exact same formatting (one word per one line). There is an additional header that is used which is keyword id (although that is not used at all right now).  
   However, the intention behind creating two separate file names was to disambiguate the fact that keywords.csv was first made to take one random keyword per audio segment for all TED audio files and llink to MSWC dataset, and the other tries to get and link all keywords per audio segment to MSWC dataset.

  To avoid ambiguity, the naming of the file keyphrases.csv may change.
  

