"""Utilidades comunes para cargar audio y separar datos por hablante."""

from pathlib import Path

import librosa
import numpy as np

SR = 16_000


def load_audio(path, dur=3.0):
    """Carga audio mono a SR y lo recorta o rellena hasta una duración fija."""
    y, _ = librosa.load(Path(path), sr=SR, mono=True)
    n = int(round(dur * SR))
    if len(y) >= n:
        return y[:n].astype("float32", copy=False)
    return np.pad(y, (0, n - len(y))).astype("float32", copy=False)


def split_by_speaker(speakers, seed=42, train=0.70, val=0.15):
    """Devuelve índices train/val/test sin compartir hablantes entre particiones."""
    speakers = np.asarray(speakers)
    unique = np.unique(speakers)
    if len(unique) < 3:
        raise ValueError(
            "Se necesitan al menos 3 hablantes distintos para separar train/val/test."
        )
    if not 0 < train < 1 or not 0 < val < 1 or train + val >= 1:
        raise ValueError("Las proporciones train y val deben ser positivas y sumar menos de 1.")

    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    n_train = min(max(1, round(len(unique) * train)), len(unique) - 2)
    n_val = min(max(1, round(len(unique) * val)), len(unique) - n_train - 1)
    train_speakers = unique[:n_train]
    val_speakers = unique[n_train:n_train + n_val]
    test_speakers = unique[n_train + n_val:]

    return tuple(
        np.flatnonzero(np.isin(speakers, group))
        for group in (train_speakers, val_speakers, test_speakers)
    )
