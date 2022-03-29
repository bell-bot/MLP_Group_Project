"""Script to run the longest audio sample"""

from src.datasets import CTRLF_DatasetWrapper
from tqdm import tqdm


x = CTRLF_DatasetWrapper()
longest_sample = 0
for i in tqdm(range(len(x.TED_sampleids_in_labels_set))):
    TED_sample_dict = x.TED.__getitem__(i)
    length = len(TED_sample_dict["waveform"][0])
    if length > longest_sample:
        longest_sample = length
        print(longest_sample)
with open("longest_sample.txt", "w") as f:
    f.write(longest_sample)