import deepspeech_pretrained as dsp
from src.datasets import CTRLF_DatasetWrapper
from moviepy.audio.AudioClip import AudioArrayClip
from scipy.io import wavfile # to read and write audio files
import os
import IPython

from src.constants import DATASET_TEDLIUM_PATH, DATASET_MLCOMMONS_PATH
from src.utils import window_split

def main():

    # Iterate through the Tedlium dataset and retrieve the necessary information
    wrapper = CTRLF_DatasetWrapper()
    accuracy = []

    # Iterate through 5000 samples
    for i in range(100):

        sample = wrapper.get(i)

        # Retrieve information from dataframe
        # We require the ted audio waveform, keyword waveform, audio start and end time, and keyword start and 
        # end time
        ted_waveform = sample["TED_waveform"][0]
        ted_waveform = ted_waveform.reshape(ted_waveform.shape[1],1)
        keyword = sample["MSWC_audio_waveform"][0]
        keyword = keyword.reshape(keyword.shape[0],1)
        ted_start_time = sample["TED_start_time"][0]
        ted_end_time = sample["TED_end_time"][0]
        ted_length = ted_end_time-ted_start_time
        ted_sample_rate = sample["TED_sample_rate"][0]
        keyword_sample_rate = sample["MSWC_sample_rate"][0]
        keyword_start_time = sample["keyword_start_time"][0]
        keyword_end_time = sample["keyword_end_time"][0]
        ted_transcript = sample["TED_transcript"][0]

        keyword_id = sample["MSWC_ID"][0]

        # Convert keyword to wav
        # Define a file to temporarly store the original audio in
        keyword_filename = "keyword_temp.wav"
        wav_file = AudioArrayClip(keyword, fps = 44100)
        wav_file.write_audiofile(keyword_filename)

        # Step 1: transcribe the keyword
        keyword_transcript, keyword_timestamps = dsp.main("deepspeech-0.9.3-models.pbmm", keyword_filename, scorer="deepspeech-0.9.3-models.scorer")

        #keyword_transcript = keyword_transcript.strip()

        """
        # Step 2: Window the data s.t. we have windows of the same length as the keyword
        frames, window_frames = window_split(actual_keyword_len, actual_keyword_len//100, ted_waveform)
        window_frames = window_frames[0].T

        ted_transcripts = []

        # Step 3: Transcribe every window
        frame_counter = 0
        for frame in window_frames:
            ted_filename =  "ted_temp.wav"
            frame = frame.reshape(frame.shape[0],1)
            wav_file = AudioArrayClip(frame, fps=44100)
            wav_file.write_audiofile(ted_filename)
            transcript = dsp.main("deepspeech-0.9.3-models.pbmm", ted_filename, scorer="deepspeech-0.9.3-models.scorer")

            ted_transcripts.append(transcript + [frame_counter])
            frame_counter += 1

        for (transcript, start, end, frame_index) in ted_transcripts:
            if transcript == keyword_transcript[0]:
                print(transcript, ted_start_time + frame_index*)
        """
        ted_filename =  "ted_temp.wav"
        wav_file = AudioArrayClip(ted_waveform, fps=44100)
        wav_file.write_audiofile(ted_filename)
        ted_transcripts, ted_timestamps = dsp.main("deepspeech-0.9.3-models.pbmm", os.path.join(DATASET_TEDLIUM_PATH, "TEDLIUM_wav", sample["TED_talk_id"][0]+".sph.wav"), scorer="deepspeech-0.9.3-models.scorer")
        keyword_len = len(keyword_transcript)
        i = 0

        print(f"Trying to identify the keyword:++{keyword_transcript}++ within transcript:++{ted_transcripts}++.")


        # Match the transcripts
        while True:
            try:
                transcript_snippet = str(ted_transcripts[i:i+keyword_len])
                if transcript_snippet == keyword_transcript:
                    ted_start = convert_timestamp(44100, ted_sample_rate, ted_start_time)
                    start = ted_timestamps[i]
                    ted_end = convert_timestamp(44100, ted_sample_rate, ted_end_time)
                    end = ted_timestamps[i+keyword_len]

                    keyword_start = convert_timestamp(44100, keyword_sample_rate, keyword_start_time)
                    keyword_end = convert_timestamp(44100, keyword_sample_rate, keyword_end_time)
                    print(start,end)
                    print(keyword_start, keyword_end)

                    start_frame_1 = int(start*16000)
                    start_frame_2 = int(keyword_start*16000)
                    end_frame_1 = int(end*16000)
                    end_frame_2 = int(keyword_end*16000)
                    print(start_frame_1, start_frame_2, end_frame_1, end_frame_2)

                    file_1 = AudioArrayClip(ted_waveform[start_frame_1:end_frame_1], fps=44100)
                    file_1.write_audiofile("file_1.wav")

                    file_2 = AudioArrayClip(ted_waveform[start_frame_2:end_frame_2], fps=44100)
                    file_2.write_audiofile("file_2.wav")
                    accuracy.append(correct_timestamp(start, end, keyword_start_time, keyword_end_time))
                    break
                i += 1
                if i > len(ted_transcript):
                    accuracy.append(False)
                    break
            except:
                break
            
        break
    #print("Model accuracy: ", sum(accuracy)/len(accuracy))
    #print("Model Done.")

def convert_timestamp(new_sr, old_sr, timestamp):
    return (new_sr*timestamp)/old_sr

def correct_timestamp(start, end, keyword_start, keyword_end):
    start_before = keyword_start <= start and keyword_end > end 
    end_after = keyword_start < start and keyword_end > end and keyword_end < end

    return start_before or end_after

main()
# y hamster has 8gb ram

