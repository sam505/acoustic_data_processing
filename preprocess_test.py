import numpy as np
from preprocess import *


def test_load_audio():
    file_path = "data/nyeri-highway/2021-09-24-16-33-27.wav"

    signal = load_audio(file_path)

    assert type(signal) == np.ndarray


def test_missing_file():
    file_path = "data/nyeri-highway/33-27.wav"

    try:
        signal = load_audio(file_path)
    except FileNotFoundError:
        assert FileNotFoundError

