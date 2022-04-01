import deepspeech_pretrained as dsp
from src.datasets import CTRLF_DatasetWrapper
from moviepy.audio.AudioClip import AudioArrayClip
from scipy.io import wavfile # to read and write audio files
import os
from tqdm import tqdm
import tensorflow as tf
import IPython
import time

from src.constants import DATASET_TEDLIUM_PATH, DATASET_MLCOMMONS_PATH

ted_filename  = "ted_temp.wav"
keyword_filename = "keyword_temp.wav"

def main():
    wrapper = CTRLF_DatasetWrapper()
    samples_to_write= sorted(list(wrapper.TED_sampleids_in_labels_set))
    
    accuracy = []
    runtime = []
    ted_lengths = {}
    acc_file = open("test_acc.txt", "w")
    run_file = open("runtime.txt", "w")

    for i in tqdm(range(len(samples_to_write))):
        try:
            runtime_per_ted = []
            start_timer = time.perf_counter()
            idx = samples_to_write[i]
            print(f"Analysing sample: {idx}")
            # Retrieve the sample at the current index
            sample = wrapper.get(idx)
            keyword_start_times = sample["keyword_start_time"].tolist()
            keyword_end_times = sample["keyword_end_time"].tolist()

            # Retrieve the ted waveform at the current index
            ted_sample_dict = wrapper.TED.__getitem__(idx)
            ted_waveform = ted_sample_dict["waveform"]
            ted_sample_rate = ted_sample_dict["sample_rate"]
            ted_start_time = ted_sample_dict["start_time"]
            ted_end_time = ted_sample_dict["end_time"]
            ted_length = ted_end_time-ted_start_time
            
            # Convert ted waveform to wav and save
            ted_waveform = ted_waveform.reshape(ted_waveform.shape[1],1)
            encode = tf.audio.encode_wav(
            ted_waveform, ted_sample_rate, name=None
            )
            tf.io.write_file(
            ted_filename, encode, name=None
            )


            # Transcribe the ted sample
            ted_transcript, ted_timestamps = dsp.main("deepspeech-0.9.3-models.pbmm", ted_filename, scorer="deepspeech-0.9.3-models.scorer")

            # Get the keyword ids for all keywords
            keywords = sample["MSWC_ID"].tolist()

            # Iterate over all keywords trying to detect them
            iterator = 0
            for keyword in keywords:
                # Retrieve keyword and information
                keyword = wrapper.MSWC.__getitem__(keyword)
                keyword_waveform = keyword["waveform"]
                keyword_waveform = keyword_waveform.reshape(keyword_waveform.shape[0],1)
                keyword_sample_rate = keyword["sample_rate"]
                keyword_start_time = keyword_start_times[iterator]
                keyword_end_time = keyword_end_times[iterator]
                iterator += 1

                # Save to wav file
                encode = tf.audio.encode_wav(
                keyword_waveform, keyword_sample_rate, name=None
                )
                tf.io.write_file(
                keyword_filename, encode, name=None
                )
            
                # transcribe the keyword
                keyword_transcript, keyword_timestamps = dsp.main("deepspeech-0.9.3-models.pbmm", keyword_filename, scorer="deepspeech-0.9.3-models.scorer")

                #keyword_transcript = keyword_transcript.strip()
            
                keyword_len = len(keyword_transcript)
                i = 0

                print(f"Trying to identify the keyword:++{keyword_transcript}++ within transcript:++{ted_transcript}++.")


                # Match the transcripts
                while True:
                    try:
                        transcript_snippet = str(ted_transcript[i:i+keyword_len])
                        if transcript_snippet == keyword_transcript:
                            start = ted_start_time+ted_timestamps[i]
                            end = ted_start_time+ted_timestamps[i+keyword_len]
                            print(start,end,keyword_start_time, keyword_end_time)
                            accuracy.append(correct_timestamp(start, end, keyword_start_time, keyword_end_time))
                            acc_file.write(str(correct_timestamp(start, end, keyword_start_time, keyword_end_time)) + "\n")
                            break
                        i += 1
                        if i > len(ted_transcript):
                            accuracy.append(False)
                            acc.write("False\n")
                            break
                    except:
                        break
                
                end_timer = time.perf_counter()
                runtime.append(end_timer-start_timer)
                run_file.write(f"{ted_length},{end_timer-start_timer}\n")
                runtime_per_ted.append(end_timer-start_timer)
                start_timer = end_timer
                if ted_length in ted_lengths.keys():
                    runtimes = ted_lengths.get(ted_length)
                    ted_lengths[ted_length] = runtimes + runtime_per_ted
        except:
            continue
            
            

    final_accuracy = sum(accuracy)/len(accuracy)
    avg_runtime = sum(runtime)/len(runtime)
    acc = open("pretrained_stats.txt", "w")
    runtimes = open("pretrained_runtimes.csv", "w")
    runtimes.write("TED_Length,Runtimes\n")
    acc.writelines([str(final_accuracy), avg_runtime])
    for (key,value) in ted_lengths.items():
        runtimes.write(f"{key},{value}\n")
    runtimes.close()
    acc.close()

    print("Model accuracy: ", sum(accuracy)/len(accuracy))
    print("Average runtime per prediction:", avg_runtime)
    print("Model Done.")

def convert_timestamp(new_sr, old_sr, timestamp):
    return (new_sr*timestamp)/old_sr

def correct_timestamp(start, end, keyword_start, keyword_end):
    start_before = keyword_start <= start and keyword_end >= end 
    end_after = keyword_end >= start and keyword_end <= end

    return start_before or end_after

main()
# y hamster has 8gb ram

