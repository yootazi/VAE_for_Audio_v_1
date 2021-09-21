#installing tensorflow 2.3.1
!pip install tensorflow==2.3.1
import tensorflow as tf
print(tf.__version__)


# installing tensorflow and torchaudio 
import tensorflow as tf
!pip install soundfile                                      #to save wav files
!pip install --no-deps torchaudio==0.5
!pip install git+https://github.com/pvigier/perlin-numpy    #for generating perlin and fractal noise


# importing libraries
import os
import pickle
import os.path
from os import path
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf


# libraries for extracting features, plotting and analysing data
from glob import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import linspace
import soundfile as sf             
import time
import IPython
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)