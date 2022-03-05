
#!/usr/bin/env python

# Source code: 
"""
Forced Alignment with Wav2Vec2

CTC segmentation algorithm described in: 
    `CTC-Segmentation of Large Corpora for German End-to-end Speech
    Recognition <https://arxiv.org/abs/2007.09127>`
Main code Modified from: `https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html`
"""

from collections import defaultdict
import csv
import os
from dataclasses import dataclass
import subprocess
import pandas as pd
import torch
import torchaudio
import re
import requests
import matplotlib
import matplotlib.pyplot as plt
# import IPython

from src.datasets import CTRLF_DatasetWrapper, KEYWORDS_LINK_CSV_PATH, TEDLIUMCustom, MultiLingualSpokenWordsEnglish, LabelsCSVHeaders, KeywordsCSVHeaders, LABELS_LINK_CSV_PATH
import link_utils

#Pre-configurations
matplotlib.rcParams['figure.figsize'] = [16.0, 4.8]
torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

# Used to Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

#TODO: Provide option to find keywords after aligning. 
class Aligner:
    PATH_TO_LABELS = LABELS_LINK_CSV_PATH
    PATH_TO_KEYWORDS = KEYWORDS_LINK_CSV_PATH
    def __init__(self, path_to_keywords=PATH_TO_KEYWORDS, overwrite=False):
        print("Preparing Ted Dataset...")
        self.TED = TEDLIUMCustom()
        print("Preparing MSWC Dataset...")
        self.MSWC = MultiLingualSpokenWordsEnglish(read_splits_file=False)
        self.keywords_df =None
        try:
            print(f"Reading {KEYWORDS_LINK_CSV_PATH}... ")
            self.keywords_df = pd.read_csv(self.PATH_TO_KEYWORDS)
        except FileNotFoundError:
            print("----- KEYWORDS CSV NOT FOUND ----- ")
            # print("-- Keywords will be generated amid aligning timestamps --")

        #Import necessary packages and the labels (characters + <UNK>)
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
        self.model = self.bundle.get_model().to(device)
        self.char_labels = self.bundle.get_labels()
        
    
 
    # ----------------  Main function ------------------- #

    #TODO!: Current function only works with SINGLE KEYWORDS. 
    def align(self):
        iterator_samples = None
        if self.keywords_df is not None:
            iterator_samples = self.keywords_df[KeywordsCSVHeaders.TED_SAMPLE_ID]
        else:
            iterator_samples = iter(range(0,self.TED.__len__()))
        with open(self.PATH_TO_LABELS, "w") as label_file:
            label_w = csv.writer(label_file)
            label_w.writerow(LabelsCSVHeaders.CSV_header)
            prev_id = None
            sample_timestamp, TED_sample_dict = None, None
            for id in iterator_samples:
                if prev_id == None or prev_id != id:
                    prev_id = id
                    TED_sample_dict = self.TED.__getitem__(id)
                    sample_timestamp = self.align_current_audio_chunk(TED_sample_dict)
                
                if self.keywords_df is not None:
                    assigned_keywords_rows = self.keywords_df[self.keywords_df[KeywordsCSVHeaders.TED_SAMPLE_ID] == id]
                    #TODO! Handle more than a single keyword
                    for key in assigned_keywords_rows[KeywordsCSVHeaders.KEYWORD]:
                        print(assigned_keywords_rows)
                        #Find Timestamps
                        split_words = key.split()
                        first_word = split_words[0]
                        last_word = split_words[-1]
                        timestamp_start = sample_timestamp[first_word][0]["start"]
                        timestamp_end = sample_timestamp[last_word][-1]["end"]
                        confidence = sample_timestamp[first_word][0]["confidence"] #TODO! Handle confidence scores of keyphrases (take minimum)
                        # confidence = 1.0
                        # for k in split_words:
                        #     current_word_timestamps = sample_timestamp[k]
                        #     for stamp in current_word_timestamps:
                        #
                        # confidence =  map()
                        ted_set, mswc_id = assigned_keywords_rows[KeywordsCSVHeaders.TED_DATASET_TYPE].values[0],  assigned_keywords_rows[KeywordsCSVHeaders.MSWC_ID].values[0]
                        label_w.writerow([key,id, ted_set, mswc_id, timestamp_start, timestamp_end, confidence])
                else:
                    #TODO! Align on the go...
                    pass



    def align_current_audio_chunk(self, TED_sample_dict):
        transcript = self.tokenise_transcript(TED_sample_dict["transcript"])
        emission = self.estimate_frame_wise_label_probability(TED_sample_dict)
        tokens, trellis = self.generate_alignment_probability(emission, transcript)
        path = self.backtrack(trellis=trellis, emission=emission, tokens=tokens)
        segments = self.merge_repeats(path, transcript)
        word_segments = self.merge_words(segments)
        word_timestamps = defaultdict(list)
        for wordS in word_segments:
            timestamp = self.get_timestamp(waveform=torch.from_numpy(TED_sample_dict["waveform"]), Segment_word=wordS, trellis=trellis)
            word = self.revert_tokenisation_process(wordS.label)

            if word_timestamps.get(word) == None:
                word_timestamps[word] = []
            word_timestamps[word].append(timestamp)
        return word_timestamps


    def get_timestamp(self, waveform, Segment_word, trellis):
        ratio = waveform.size(1) / (trellis.size(0) - 1)
        x0 = int(ratio * Segment_word.start)
        x1 = int(ratio * Segment_word.end)
        print(f"{Segment_word.label} ({Segment_word.score:.2f}): {x0 / self.bundle.sample_rate:.3f} - {x1 / self.bundle.sample_rate:.3f} sec")
        timestamp = {
            "start": x0 / self.bundle.sample_rate , 
            "end": x1 / self.bundle.sample_rate, 
            "confidence": Segment_word.score
        }
        return timestamp

    
    def pick_keyword():
        pass

    # ----------------  Helper functions ------------------- #

    # ------ Tokenisation ----- #
    #Helper function to turn transcript into tokens of characters for the Wav2Vec2
    def tokenise_transcript(self, transcript_original):
        transcript_original = link_utils.preprocess_text(transcript_original)
        transcript_preprocessing = transcript_original.upper().replace("<UNK>","<unk>").strip().split() 
        transcript = ["<s>"]
        for idx, word in enumerate(transcript_preprocessing):
            if word == "<unk>":
                transcript.append(word)
            else:
                
                word = link_utils.parse_number_string(word)
                word = word.upper() #ensure word is turned to upper case

                for c in word: 
                    transcript.append(c)
            if idx != len(transcript_preprocessing) -1:
                transcript.append("|")
            else:
                transcript.append("</s>")
        print(transcript)
        return transcript

    def revert_tokenisation_process(self, word):
        if "<s>" in word:
            word = word.replace("<s>", "")
        elif "</s>" in word:
            word = word.replace("</s>", "")
        else:
            word = word
        
        return word.lower()

    # ------ Miscellaneous ----- #

    #Reads from CSV file the last read sample id, to continue from last time we stopped
    def retrieve_last_sample_id(self):
        ted_sample_id_column = 1
        #NOTE: Assert that the order is the same, though not a robust solution, it was done. The additional assertion check is done to make sure we are not reading from another column
        assert(LabelsCSVHeaders.TED_SAMPLE_ID == LabelsCSVHeaders.CSV_HEADER[ted_sample_id_column])
        #subprocess requires byte like object to read
        line = subprocess.check_output(['tail', '-1', bytes(self.KEYWORDS_LINK_FILENAME, encoding="utf-8")])
        print(line)
        #Convert from bytes to int 
        last_read_sample_id = int(str(line.split(b',')[ted_sample_id_column], encoding="utf-8"))
        last_read_sample_id = last_read_sample_id + 1 #We will start iterating from the sample id after the last one
        print(f"Side note: The python script assumes that the TED Sample ID column is in {ted_sample_id_column}")
        print(f"Last TED Sample ID Read: {last_read_sample_id}")
        return last_read_sample_id

# -------------------------- Generate frame-wise label probability --------------------- #

    ######################################################################
    # Generate frame-wise label probability
    # -------------------------------------
    # 
    # The first step is to generate the label class porbability of each aduio
    # frame. We can use a Wav2Vec2 model that is trained for ASR. Here we use
    # :py:func:`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`.
    # 
    # ``torchaudio`` provides easy access to pretrained models with associated
    # labels.
    # 
    # .. note::
    #
    #    In the subsequent sections, we will compute the probability in
    #    log-domain to avoid numerical instability. For this purpose, we
    #    normalize the ``emission`` with :py:func:`torch.log_softmax`.
    # 
    def estimate_frame_wise_label_probability(self, TED_sample_dict):
        with torch.inference_mode():
        #   waveform, _ = torchaudio.load(SPEECH_FILE)
            waveform = torch.from_numpy(TED_sample_dict["waveform"])
            emissions, _ = self.model(waveform.to(device))
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()
        return emission

# -------------------------- Generate Alignment Probability (trellis) --------------------- #
    ######################################################################
    # Generate alignment probability (trellis)
    # ----------------------------------------
    # 
    # From the emission matrix, next we generate the trellis which represents
    # the probability of transcript labels occur at each time frame.
    # 
    # Trellis is 2D matrix with time axis and label axis. The label axis
    # represents the transcript that we are aligning. In the following, we use
    # :math:`t` to denote the index in time axis and :math:`j` to denote the
    # index in label axis. :math:`c_j` represents the label at label index
    # :math:`j`.
    # 
    # To generate, the probability of time step :math:`t+1`, we look at the
    # trellis from time step :math:`t` and emission at time step :math:`t+1`.
    # There are two path to reach to time step :math:`t+1` with label
    # :math:`c_{j+1}`. The first one is the case where the label was
    # :math:`c_{j+1}` at :math:`t` and there was no label change from
    # :math:`t` to :math:`t+1`. The other case is where the label was
    # :math:`c_j` at :math:`t` and it transitioned to the next label
    # :math:`c_{j+1}` at :math:`t+1`.
    # 
    # The follwoing diagram illustrates this transition.
    # 
    # .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/ctc-forward.png
    # 
    # Since we are looking for the most likely transitions, we take the more
    # likely path for the value of :math:`k_{(t+1, j+1)}`, that is
    # 
    # :math:`k_{(t+1, j+1)} = max( k_{(t, j)} p(t+1, c_{j+1}), k_{(t, j+1)} p(t+1, repeat) )`
    # 
    # where :math:`k` represents is trellis matrix, and :math:`p(t, c_j)`
    # represents the probability of label :math:`c_j` at time step :math:`t`.
    # :math:`repeat` represents the blank token from CTC formulation. (For the
    # detail of CTC algorithm, please refer to the *Sequence Modeling with CTC*
    # [`distill.pub <https://distill.pub/2017/ctc/>`__])
    # 



    def generate_alignment_probability(self, emission, transcript ):
        tokens = self.get_tokens(transcript)
        trellis = self.get_trellis(emission, tokens)
        return tokens, trellis

    def get_tokens(self, transcript):
        dictionary  = {c: i for i, c in enumerate(self.char_labels)}
        print(dictionary)
        tokens = [dictionary[c] for c in transcript]
        # print(list(zip(transcript, tokens)))
        return tokens


    def get_trellis(self,emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code. 
        trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
        trellis[:, 0] = 0
        for t in range(num_frame):
            trellis[t+1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

 

#####################################################################
# Find the most likely path (backtracking)
# ----------------------------------------
# 
# Once the trellis is generated, we will traverse it following the
# elements with high probability.
# 
# We will start from the last label index with the time step of highest
# probability, then, we traverse back in time, picking stay
# (:math:`c_j \rightarrow c_j`) or transition
# (:math:`c_j \rightarrow c_{j+1}`), based on the post-transition
# probability :math:`k_{t, j} p(t+1, c_{j+1})` or
# :math:`k_{t, j+1} p(t+1, repeat)`.
# 
# Transition is done once the label reaches the beginning.
# 
# The trellis matrix is used for path-finding, but for the final
# probability of each segment, we take the frame-wise probability from
# emission matrix.
# 




    def backtrack(self, trellis, emission, tokens, blank_id=0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions 
        # for time and tokens at the beginning.
        # When refering to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when refering to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t-1, j] + emission[t-1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]

            # 2. Store the path with frame-wise probability.
            prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j-1, t-1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError('Failed to align')
        return path[::-1]


################################################################################
# Visualization
################################################################################
    def plot_trellis_with_path(self, trellis, path):
        # To plot trellis with path, we take advantage of 'nan' value
        trellis_with_path = trellis.clone()
        for i, p in enumerate(path):
            trellis_with_path[p.time_index, p.token_index] = float('nan')
        plt.imshow(trellis_with_path[1:, 1:].T, origin='lower')
        plt.title("The path found by backtracking")
        plt.show()

    # plot_trellis_with_path(trellis, path)
    # plt.title("The path found by backtracking")
    # plt.show()

    
    ######################################################################
    # Looking good. Now this path contains repetations for the same labels, so
    # let’s merge them to make it close to the original transcript.
    # 
    # When merging the multiple path points, we simply take the average
    # probability for the merged segments.
    # 

    # Merge the labels
    def merge_repeats(self, path, transcript):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2-1].time_index + 1, score))
            i1 = i2
        return segments


    ######################################################################
    # Looks good. Now let’s merge the words. The Wav2Vec2 model uses ``'|'``
    # as the word boundary, so we merge the segments before each occurance of
    # ``'|'``.
    # 
    # Then, finally, we segment the original audio into segmented audio and
    # listen to them to see if the segmentation is correct.
    # 
    def merge_words(self,segments, separator='|'):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs= segments[i1:i2]
                    #IF the next character after the separator (if any) is a punctuation, we will have to merge the entire word that follows after the seperator as a whole
                    #eg. ["D", "O", "|", "'",  "N", "T"] should become "DON'T" instead of being separated as "DON" and "'T"
                    if i2 +1 < len(segments) -1 and segments[i2].label == separator and segments[i2+1].label == '\'':
                        i2 = i2+1
                        i_punc = i2  
                        while i2 < len(segments) and segments[i2].label!=separator:
                            i2+=1
                            
                        segs = segs +  segments[i_punc:i2]
  
                    word = ''.join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2-1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words



################################################################################
# Visualization
################################################################################
    def plot_alignments(self, trellis, segments, word_segments, waveform):
        trellis_with_path = trellis.clone()
        for i, seg in enumerate(segments):
            if seg.label != '|':
                trellis_with_path[seg.start+1:seg.end+1, i+1] = float('nan')

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))

        ax1.imshow(trellis_with_path[1:, 1:].T, origin='lower')
        ax1.set_xticks([])
        ax1.set_yticks([])

        for word in word_segments:
            ax1.axvline(word.start - 0.5)
            ax1.axvline(word.end - 0.5)

        for i, seg in enumerate(segments):
            if seg.label != '|':
                ax1.annotate(seg.label, (seg.start, i + 0.3))
                ax1.annotate(f'{seg.score:.2f}', (seg.start , i + 4), fontsize=8)

        # The original waveform
        ratio = waveform.size(0) / (trellis.size(0) - 1)
        ax2.plot(waveform)
        for word in word_segments:
            x0 = ratio * word.start
            x1 = ratio * word.end
            ax2.axvspan(x0, x1, alpha=0.1, color='red')
            ax2.annotate(f'{word.score:.2f}', (x0, 0.8))

        for seg in segments:
            if seg.label != '|':
                ax2.annotate(seg.label, (seg.start * ratio, 0.9))
        xticks = ax2.get_xticks()
        plt.xticks(xticks, xticks / self.bundle.sample_rate)
        ax2.set_xlabel('time [second]')
        ax2.set_yticks([])
        ax2.set_ylim(-1.0, 1.0)
        ax2.set_xlim(0, waveform.size(-1))





# A trick to embed the resulting audio to the generated file.
# `IPython.display.Audio` has to be the last call in a cell,
# and there should be only one call par cell.
    def display_segment(self,i, waveform, trellis, word_segments):
        ratio = waveform.size(1) / (trellis.size(0) - 1)
        word = word_segments[i]
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        filename = f"_assets/{i}_{word.label}.wav"
        torchaudio.save(filename, waveform[:, x0:x1], self.bundle.sample_rate)
        print(f"{word.label} ({word.score:.2f}): {x0 / self.bundle.sample_rate:.3f} - {x1 / self.bundle.sample_rate:.3f} sec")
        return IPython.display.Audio(filename)

######################################################################
# 

from link import KeywordLinker

if __name__=="__main__":
    print(torch.__version__)
    print(torchaudio.__version__)
    print(device)
    #Change working directory to where the script is
    os.chdir(os.path.abspath(os.path.dirname(__file__))) 
 
    AlignerEngine = Aligner()
    AlignerEngine.align()


    ######### Testing Aligner ############
    # x = Aligner()
    # TED_sample_dict = x.TED.__getitem__(17)
    # sample_timestamps = x.align_current_audio_chunk(TED_sample_dict)
    # print(sample_timestamps)