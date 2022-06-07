from random import sample
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
import moviepy.editor as mp
import librosa

from pydub import AudioSegment

##Inspired by code from
#https://iq.opengenus.org/mfcc-audio/
#and
#https://publish.illinois.edu/augmentedlistening/tutorials/music-processing/tutorial-1-introduction-to-audio-processing-in-python/
##There are exmaples present internaly and variables are self describing.
# ##The MP4 and WAV files are stored in the same place as main.py (needs to be renamed in the future) or specifcy paths in strings




####################Single files#################################################################
def read_WAV_file(wave):
    '''
    reads the mp4 file as data and frequency
    :param wave (string): needs to be in form "Shrek.wav" and reads in data and frequency
    :return None: sampleing frequncy, data in 1d form
    '''
    fs, data = read(wave) # fs is the sampling frequency and data is the read in audio file
    data = data[:, 0]   # reshape to make it 1d
    return fs, data


def read_MP4_file(mp4_path, output_file):
    '''
    reads the mp4 file and wrties to desired format
    :param mp4_path (string): needs to be in form -r"Shrek.mp4"- as r indicates read
    :param output_file (string): needs to be in form -r"Shrek.wav"- as r indicates read
    :return None: writes file
    '''
    video = mp.VideoFileClip(mp4_path)  # here sample.mp4 is the name of video clip. 'r' indicates that we are
    # reading a file
    video.audio.write_audiofile(output_file)  # Here output_file is the name of the audio file with type e.g "Shrek.wav"

    return video.audio.to_soundarray()

def play_WAV_audio(fs, data):
    '''
    plays the audio file
    :param fs (int): sampleling frequency in hz
    :param data (1d array): aduio data in 1d form
    :return None: plays audio
    '''
    return Audio(data, rate=fs)    #plays the audio

# From https://github.com/jiaaro/pydub/blob/master/API.markdown
def read_audio(file, extension):
    '''
    reads media files into audio and returns numpy array and frame rate
    :param file : file path to the media file
    :param extension: extension of the file to be read. Example, to read mp4 files, provide "mp4" to the extension. 
    return numpy array and frame rate (sampling frequency in hz)
    '''
    sound = AudioSegment.from_file(file, format=extension)
    channel_sounds = sound.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return sound.frame_rate, fp_arr


def write_WAV_audio(fs, data, file_name):
    '''
    Plots the audio file as a waveform
    :param data (1d array): aduio data in 1d form
    :return None: writes file
    '''
    write(file_name, fs, data)  #writes to a new file (for generation maybe?)


def plot_WAV_audio(data):
    '''
    Plots the audio file as a waveform
    :param data (1d array): aduio data in 1d form
    :return None: shows plot
    '''
    plt.figure()
    plt.plot(data)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Waveform of Audio")
    plt.show()

###############################################################################################

###########################Multiple files and Pre Processing###################################
RATE = 24000        ##Sampling rate in hz
N_MFCC = 13         ##Number of MFCC samples to take

def get_wav(language_num):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''
    y, sr = librosa.load('./{}.wav'.format(language_num)) #Make sure to have audio file in your desktop or you may change the path as per your need
    return resample_audio(data=y, sampling_rate=sr)

def resample_audio(data, sampling_rate, target_rate=RATE):
    return(librosa.core.resample(y=data,orig_sr=sampling_rate,target_sr=RATE, scale=True))


def to_mfcc(wav, target_rate=RATE, n_mfcc=N_MFCC):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def scale_mfcc(mfcc_features):
    return np.mean(mfcc_features.T, axis=0)

def extract_mfcc_features_from_waveform(audio, sample_rate, num_mfcc=40):
    mfccs_features = to_mfcc(wav=audio, target_rate=sample_rate, n_mfcc=num_mfcc)
    mfccs_scaled_features= scale_mfcc(mfcc_features=mfccs_features)
    return mfccs_scaled_features
    
###############################################################################################

###########################Multiple Files in local dir folder##################################
def preprocess_files(list_of_audio_filenames):
    '''
    Does preprocessing steps for multiple files
    :param list_of_audio_filenames (list): list of JUST names of files ("Shrek" not "Shrek.wav")
    :return None: Writes each audio to a txt file containing the MFCC's of the data
    '''
    if len(list_of_audio_filenames) == 0:
        print("Empty list")
        exit()

    for a in list_of_audio_filenames:
        wave = get_wav(a)
        mfcc = to_mfcc(wave)

        file_name = a+".txt"
        c = np.savetxt(file_name, mfcc, delimiter=', ')
        a = open(file_name, 'r')

    # Press the green button in the gutter to run the script.

def preprocess(fs, data, target_rate=RATE, n_mfcc= N_MFCC):
    audio = resample_audio(data=data,sampling_rate=fs, target_rate=target_rate)
    audio = to_mfcc(audio, target_rate, n_mfcc)
    # return audio.reshape((audio.shape[0],audio.shape[1]))
    return audio

if __name__ == '__main__':
    list_of_files = list()
    list_of_files.append("Shrek")
    preprocess_files(list_of_files)
    print("Done!")

    ##EXAMPLES######################################
    #Reading in a single file and converting to mp3#
    # read_MP4_file(r"Shrek.mp4", r"Shrek.wav")
    #fs, data = read_WAV_file("Shrek.wav")
    #play_WAV_audio(fs, data)
    #plot_WAV_audio(data)
    ################################################

    ###Pre Procesisng for single example############
    #audio_file = "Shrek"

    #wave = get_wav(audio_file)
    #print(wave.shape)
    #mfcc = to_mfcc(wave)
    #print(mfcc.shape)

    #c = np.savetxt('file.txt', mfcc, delimiter =', ')
    #a = open("file.txt", 'r')                                       # open file in read mode
    #print("Done!")
    ################################################
    ################################################