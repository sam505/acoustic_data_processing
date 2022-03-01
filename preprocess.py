import numpy as np
import librosa
import configparser
import warnings
import os

warnings.filterwarnings('ignore')


# Get parameters from configuration file
config = configparser.ConfigParser()
config.read('files/parameters-birds.ini')

win_len_ms = int(config['audio']['win_len_ms'])
overlap = float(config['audio']['overlap'])
sampling_rate = int(config['audio']['sampling_rate'])
duration = float(config['neural-net']['input_duration_s'])
rnd_seed = int(config['neural-net']['seed'])
num_mels = int(config['baseline']['num_mels'])


# Derive audio processing values
win_len = int((win_len_ms * sampling_rate) / 1000)
hop_len = int(win_len * (1 - overlap))
nfft = int(2 ** np.ceil(np.log2(win_len)))
num_frame = int((0.5 * duration * sampling_rate) / hop_len)


def load_audio(audio_path):
    """

    :param audio_path:
    :return: s
    """
    try:
        s, _ = librosa.load(audio_path, sr=sampling_rate)
        return s
    except FileNotFoundError:
        print(f"{audio_path} does not exist")


audio_path = "data/nyeri-highway/2021-09-24-16-33-27.wav"
load_audio(audio_path)

