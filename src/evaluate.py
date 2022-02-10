from compare import match_audio
from Preprocessing.pre_processing import *
from Preprocessing.sliding_windows import create_sliding_windows

import pandas as pd

RECORDINGS_METADATA_PATH = './recording_examples/recordings_metadata.csv'

def read_recordings_metadata(path = RECORDINGS_METADATA_PATH):
    metadata = pd.read_csv(path)
    print(metadata)

def evaluate_samples():
    pass
def evaluate_exact_recordings():
    pass

def evaluate():
    pass

if __name__ == "__main__":
    read_recordings_metadata()
