# Code from :`https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/audio/ipynb/ctc_asr.ipynb#scrollTo=yY84aJdKPkbA``

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer
import csv
import os

from tqdm import tqdm
from src.datasets import CTRLF_DatasetWrapper
from Data import data_utils
metadata_path = "/home/szy/Documents/MLP_Group_Project/src/metadata.csv"
wav_path = "/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/Data/TEDLIUM_release-3/data/wav/"
model_directory = "/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/models/"
generated_text_stats_path = "/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/models/"
generated_history_path = "/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/src/models/"

# --------- HYPERPARAMETER SETTINGS --------------- 
REMOVE_UNK = True
PREPROCESS = True
# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384
NUM_OF_SAMPLES = 6000 #<---- DATASET SIZE
BATCH_SIZE = 16
RNN_UNITS= 128 #original: 512
RNN_LAYERS = 2 #original : 5
LR_ADAM = 1e-3

EPOCHS =100
# ------------------------------------------

print(tf.__version__)
print(tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

CTRLF_Engine = CTRLF_DatasetWrapper()

def read_dataset_metadata():
    # Read metadata file and parse it
    metadata_df = pd.read_csv(metadata_path)
    metadata_df.columns = ["TED_waveform","start_time","end_time","TED_transcript"]
    metadata_df = metadata_df[["TED_waveform","TED_transcript"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    metadata_df.head(3)
    return metadata_df

def read_ctrlf_dataset(num_of_samples=3000):
    audio_df = pd.DataFrame(columns=CTRLF_Engine.COLS_OUTPUT_TED_SIMPLIFIED)
    output_rows = []
    for i in tqdm(range(0,num_of_samples),desc="Preparing Dataset"):
        row = CTRLF_Engine.get_ted_talk_id_and_transcript(i)
        
        if len(row) ==0:
            num_of_samples+=1
            continue
        else:
            row[0]= str(i) + "_" + row[0]
            print(row[1])
            if REMOVE_UNK:
                row[1] = row[1].replace("<unk>", "") #TODO: See if this is plausible
            if PREPROCESS:
                row[1] = data_utils.preprocess_text(row[1])
                try:
                    tokens = [data_utils.parse_number_string(word) for word in row[1].split()]
                    row[1] = " ".join(tokens)
                except:
                    continue
            output_rows.append(row)
    audio_df = pd.DataFrame(data= output_rows, columns=["TED_Talk_ID", "TED_transcript"])
    audio_df.reset_index(inplace=True, drop=True)
    return audio_df
        

def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
     # 1. Read wav file
    file = tf.io.read_file(wav_path + wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=LR_ADAM)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


metadata_df = read_ctrlf_dataset(num_of_samples=NUM_OF_SAMPLES)
print(metadata_df[0:10])
split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]
print(f"Size of the training set: {len(df_train)}")
print(f"Size of the training set: {len(df_val)}")
# The set of characters accepted in the TED_transcription.

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")

# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)

# Define the trainig dataset


train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["TED_Talk_ID"]), list(df_train["TED_transcript"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["TED_Talk_ID"]), list(df_val["TED_transcript"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

#Disable Auto Sharding
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_dataset = train_dataset.with_options(options)
validation_dataset = validation_dataset.with_options(options)


# fig = plt.figure(figsize=(8, 5))
# for batch in train_dataset.take(1):
#     spectrogram = batch[0][0].numpy()
#     spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
#     label = batch[1][0]
#     # Spectrogram
#     label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
#     ax = plt.subplot(2, 1, 1)
#     ax.imshow(spectrogram, vmax=1)
#     ax.set_title(label)
#     ax.axis("off")
#     # Wav
#     file = tf.io.read_file(wav_path + list(df_train["TED_Talk_ID"])[0] + ".wav")
#     audio, _ = tf.audio.decode_wav(file)
#     audio = audio.numpy()
#     ax = plt.subplot(2, 1, 2)
#     plt.plot(audio)
#     ax.set_title("Signal Wave")
#     ax.set_xlim(0, len(audio))
#     display.display(display.Audio(np.transpose(audio), rate=16000))
# plt.show()


# -------------------------- Model ---------------------------------- #

def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=LR_ADAM)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint, custom_objects= {'CTCLoss': CTCLoss})
    print("Creating a new model")
    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=RNN_UNITS,
        rnn_layers=RNN_LAYERS
    )
    model.summary(line=110)
    return model


##Class needs to be here
# A callback class to output a few TED_transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = 
            predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)
        
        if ('lr' not in logs.keys()):
            logs.setdefault('lr',0)
            logs['lr'] = keras.backend.get_value(self.model.optimizer.lr)

        if not ('model_history.csv' in os.listdir(model_directory)):
            with open(model_directory+'model_history.csv','a') as f:
                y=csv.DictWriter(f,logs.keys())
                y.writeheader()

        with open(model_directory+'model_history.csv','a') as f:
            y=csv.DictWriter(f,logs.keys())
            y.writerow(logs)



def train_model():
    # Define the number of epochs.
    epochs = EPOCHS
    # Callback function to check TED_transcription on the val set.
    validation_callback = CallbackEval(validation_dataset)
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[validation_callback],
    )
    return history

def eval_model(history, num_of_predictions_to_print=5):
    
    # Let's check results on more validation samples
    predictions = []
    targets = []
    for batch in validation_dataset:
        X, y = batch
        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
        for label in y:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            targets.append(label)
    tf.saved_model.save(model, model_directory)
    wer_score = wer(targets, predictions)
    
    with open(model_directory + "model_stats.txt", "a") as f:  
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        f.write("WER: " + str(wer_score) +"\n")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), num_of_predictions_to_print):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)

            f.write(f"Target: {targets[i]}\n")
            f.write(f"Prediction: {predictions[i]}\n") 
    
 
        
        
#Train the model
history =train_model()
# Let's check results on more validation samples
eval_model(history)


    