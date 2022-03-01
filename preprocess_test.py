import numpy as np

from preprocess import *


def test_load_audio():
    file_path = "data/nyeri-highway/2021-09-24-16-33-27.wav"

    signal = load_audio(file_path)

    assert type(signal) == np.ndarray


def test_missing_file():
    file_path = "data/nyeri-highway/33-27.wav"

    signal = load_audio(file_path)

    assert not signal


def test_directory():
    path = "data/nyeri-highway"

    audio_arr = load_audios_from_folder(path)
    for arr in audio_arr:
        assert type(arr) == np.ndarray


def test_not_a_directory():
    path = "audio_data"

    audio_arr = load_audios_from_folder(path)
    for arr in audio_arr:
        assert type(arr) != np.ndarray
