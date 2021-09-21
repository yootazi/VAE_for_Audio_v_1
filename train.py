import os
import numpy as np


#@title Hyperparameters 

learning_rate = 0.0005 #@param {type:"raw"}
num_epochs_to_train =  40#@param {type:"integer"}
batch_size = 64 #@param {type:"integer"}
vector_dimension = 64 #@param {type:"integer"}

hop=256               #hop size (window size = 4*hop)
sr=44100              #sampling rate
min_level_db=-100     #reference values to normalize data
ref_level_db=20

LEARNING_RATE = learning_rate
BATCH_SIZE = batch_size
EPOCHS = num_epochs_to_train
VECTOR_DIM=vector_dimension

shape=128           #length of time axis of split specrograms         
spec_split=1        

SPECTROGRAMS_SAVE_DIR = '/content/gdrive/MyDrive/musicdata/vae_for_audio/spectrograms'
MIN_MAX_VALUES_SAVE_DIR = '/content/gdrive/MyDrive/musicdata/vae_for_audio'
FILES_DIR = '/content/gdrive/MyDrive/musicdata/vae_for_audio/audio/'



def load_fsdd(SPECTROGRAMS_PATH):                                      # loading spectrograms and saving it in x_train array
    x_train = []
    for root, _, file_names in os.walk(SPECTROGRAMS_SAVE_DIR):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path, allow_pickle=True)        # (n_bins, n_frames, 1)  # ->shape of the array / convolutional layers have 3 dimension array/ we should another dimension (1)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]                                 # -> (3000, 256, 64, 1)       treating spectrograms as a grayscale images
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim= 128                     #instead of 128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_SAVE_DIR)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")