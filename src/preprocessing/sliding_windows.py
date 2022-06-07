import numpy as np
import librosa



print(np.__version__)
"""
Given a flattened numpy array, transforms the data into sliding windows, and the overlap between each window can be defined by step size.
If the window_size == step_size, then there will be no overlap between the windows created. 
Examples:
1) Given data sample that is [0,1,2,3], a window size of 2 and a step size of 2 will output: [[0,1], [2,3]]. Hence, 0% overlap
2) Given data sample that is [0,1,2,3], a window size of 2 and a step size of 1 will output: [[0,1], [1,2], [2,3]].  Hence, 50% overlap.
Overlap can be defined as (window_size- step_size) / window_size
Note: Depending on the configuration of window size and step size, values at the end of the inputted data may be dropped/trimmed/removed from the final output. 

:param audio_input: The flattened numpy array to be transformed
:param window_size: The size of each sliding window
:param step_size: The overlap desired the current window (call window w) and the previous window( window w-1), and so on.
:param copy: If copy is set to False, this creates a memory-shared view (shallow copy/ reference). Changing the output returned will change the input as well. Set copy to True to return a new object (deep copy).
:return Return numpy array with sliding windows
"""

def create_sliding_windows(audio_input, window_size = 25, step_size = 25, copy = False):
    output_view = np.lib.stride_tricks.sliding_window_view(x= audio_input, window_shape = window_size)[0::step_size]
    return (output_view.copy() if copy else output_view)

#Test Sliding windows
if __name__ == "__main__": 
    #data = np.arange(1,10)
    #print("Original data shape: {}", data.shape)
    #print(data)
    #output = create_sliding_windows(data,copy=False)
    #print("New transformed data shape: {}", output.shape)
    #print(output)
    ##^^ this is not a sliding window##
    x = np.arange(0, 128)
    frame_len, hop_len = 16, 8
    frames = librosa.util.frame(x, frame_length=frame_len, hop_length=hop_len)
    windowed_frames = np.hanning(frame_len).reshape(-1, 1) * frames

    # Print frames
    for i, frame in enumerate(frames):
        print("Frame {}: {}".format(i, frame))

    # Print windowed frames
    for i, frame in enumerate(windowed_frames):
        print("Win Frame {}: {}".format(i, np.round(frame, 3)))




