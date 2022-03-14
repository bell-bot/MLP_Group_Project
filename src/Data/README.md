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

# Downloading Datasets
Please check the scripts folder for more information


# File Descriptions

## Labels.csv 
Contains keywords 
- Keyword: str
  -   The keyword linking the two audio files (sample of a TED audio file and an MSWC recording of that keyword)
- TEDLIUM_SampleID: int 
  - Represents the id of the audio segment in TED talk. In other words, it is a unique id that maps to a segment of a TED audio file. 
                    This is NOT the same as "TED_TALK_ID", which represents the id of an entire audio file. 
- TED_TALK_ID: str
  - A unique id of a TED audio file. Added in order to make referencing the current audio segment/sample to the original TED audio file more accessible
- TEDLIUM_SET: str
  - The type of dataset the audio sample/segment (or more specifically the audio file) exists in
  - It can be one of the four values: Train vs Dev vs Test vs None
  - Currently, it is not used as it is not possible to work with all the TED data. Working with our subset, we will ensure we work on Train and Dev data (ideally), split up into our own train, valid, test split.
- MSWC_AudioID: str
  - The audio id of the keyword recording from MSWC chosen to link with a TED audio segment. 
  - It is be used to load the audio from MSWC dataset.  
- start_time: float
  - The starting of the interval time of when the current keyword was said in the audio segment.
- end_time: float
  - The end of the interval time of when the current keyword was said in the audio segment.
- confidence: float
  - The average probabilty score of the forced alignment model on generating the keyword timestamp. Represents the model's frame-wise probability from emission matrix (frame-wise label probability distribution). 
  - https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
 
 
 
 Model used to generate timestamps is WAV2VEC2_ASR_BASE_960H (https://pytorch.org/audio/main/pipelines.html).


## Keywords.csv/ Keyphrases.csv
This is used as an intermediate step to create labels.csv. These contain the same headers as labels.csv, but before force alignment is applied to them (to generate timestamps).

- Side notes:
   keyphrases.csv is very similar to keywords.csv with exact same formatting (one word per one line). There is an additional header that is used which is keyword id (although that is not used at all right now).  
   However, the intention behind creating two separate file names was to disambiguate the fact that keywords.csv was first made to take one random keyword per audio segment for all TED audio files and llink to MSWC dataset, and the other tries to get and link all keywords per audio segment to MSWC dataset.

  To avoid ambiguity, the naming of the file keyphrases.csv may change
