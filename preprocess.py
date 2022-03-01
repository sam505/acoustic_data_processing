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

    :param audio_path: Path to a single audio file
    :return: s: an array representation of the audio
    """
    try:
        s, _ = librosa.load(audio_path, sr=sampling_rate)
        return s
    except FileNotFoundError:
        print(f"{audio_path} does not exist")


def load_audios_from_folder(directory):
    """

    :param directory:
    :return: audio_array
    """
    try:
        audio_array = []
        files = os.listdir(directory)
        for file in files:
            audio_array.append(load_audio(os.path.join(directory, file)))

        yield np.array(audio_array)

    except NotADirectoryError:
        print(f"{directory} does not exist")


