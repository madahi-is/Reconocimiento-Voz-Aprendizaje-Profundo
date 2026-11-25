"""Análisis prosódico y acústico de voz con aprendizaje profundo (CNN, LSTM y CNN-LSTM).

Clases (una carpeta por clase) y nombre de archivo <hablante>_<n>.wav:
  data/prosodia/habla_normal/   data/prosodia/grito/
  data/prosodia/elevacion_tonal/   data/prosodia/error_pronunciacion/

Entradas de cada arquitectura:
  CNN       -> espectrograma Mel (64 x 301)
  LSTM      -> secuencia de descriptores por frame: logF0, sonoridad, dF0, log-RMS, ZCR,
               centroide espectral y 13 MFCC  (188 frames x 19)
  CNN-LSTM  -> espectrograma Mel; la CNN extrae patrones locales, una BiLSTM con atención
               modela su evolución temporal (como la Figura 3 del informe)

La partición train/val/test es POR HABLANTE (voces no vistas en el entrenamiento).

Uso:
  python prosodia_dl.py train --data data/prosodia --arch all
  python prosodia_dl.py predict --file audio.wav --arch cnn_lstm
"""
import argparse
import hashlib
import json
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from audio_utils import SR, load_audio, split_by_speaker

DUR = 3.0                      # duración fija de cada audio (s)
N_MELS, HOP_MEL = 64, 160      # 3 s -> 301 frames Mel
HOP_SEQ = 256                  # 16 ms por frame en la secuencia de descriptores
T_SEQ = int(DUR * SR / HOP_SEQ) + 1   # 188
N_FEAT = 19


# --------------------------------------------------------------------------- features
def mel_fijo(y):
    """Mel en dB con escala FIJA. No se normaliza por muestra: la intensidad absoluta
    es información clave para distinguir un grito del habla normal."""
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, hop_length=HOP_MEL, n_mels=N_MELS)
    S = np.clip(librosa.power_to_db(S, ref=1.0, amin=1e-10), -80.0, 20.0)
    return ((S + 30.0) / 50.0)[..., None].astype("float32")


def _fijar(v, n):
    return v[:n] if len(v) >= n else np.pad(v, (0, n - len(v)))


def secuencia(y):
    """Descriptores acústico-prosódicos por frame -> (T_SEQ, N_FEAT)."""
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=500, sr=SR, frame_length=1024, hop_length=HOP_SEQ)
    voz = ~np.isnan(f0)
    logf0 = np.where(voz, np.log2(np.where(voz, f0, 100.0) / 100.0), 0.0)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP_SEQ)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=HOP_SEQ)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=1024, hop_length=HOP_SEQ)[0] / (SR / 2)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=1024, hop_length=HOP_SEQ)
    filas = [logf0, voz.astype(float), np.r_[0.0, np.diff(logf0)], np.log10(rms + 1e-5), zcr, cen]
    filas = [_fijar(f, T_SEQ) for f in filas] + [_fijar(m, T_SEQ) for m in mfcc]
    return np.stack(filas, axis=1).astype("float32")


def construir_dataset(data, cache, recomputar=False):
    raiz = Path(data).resolve()
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe el directorio de datos: {raiz}")
    archivos = sorted(raiz.rglob("*.wav"))
    huella = hashlib.sha256(Path(__file__).read_bytes())
    huella.update(Path(__file__).with_name("audio_utils.py").read_bytes())
    huella.update(str(raiz).encode())
    for archivo in archivos:
        info = archivo.stat()
        huella.update(repr((str(archivo.relative_to(raiz)), info.st_size, info.st_mtime_ns)).encode())
    clave = huella.hexdigest()
    if Path(cache).exists() and not recomputar:
        with np.load(cache) as d:
            if "clave" in d and d["clave"].item() == clave:
                print("Caracter?sticas cargadas desde la cach?.")
                return d["mel"], d["seq"], d["labels"], d["spk"]
    mel, seq, labels, spk = [], [], [], []
    if not archivos:
        raise ValueError(f"No hay carpetas de clases dentro de {raiz}")
    conteo = {}
    for i, f in enumerate(archivos, 1):
        if i == 1 or i % 50 == 0 or i == len(archivos):
            print(f"Procesando audio {i}/{len(archivos)}")
        relativo = f.relative_to(raiz)
        if len(relativo.parts) < 2:
            raise ValueError(
                f"El audio {f} debe estar dentro de una carpeta de clase (y opcionalmente una de hablante)."
            )
        clase = relativo.parts[0]
        y = load_audio(f, dur=DUR)
        mel.append(mel_fijo(y)); seq.append(secuencia(y))
        labels.append(clase); spk.append(f.stem.split("_")[0])
        conteo[clase] = conteo.get(clase, 0) + 1
    for clase, cantidad in sorted(conteo.items()):
        print(f"  {clase}: {cantidad} audios listos")
    mel, seq, labels, spk = map(np.array, (mel, seq, labels, spk))
    np.savez_compressed(cache, mel=mel, seq=seq, labels=labels, spk=spk, clave=clave)   # evita recalcular pyin
    return mel, seq, labels, spk


# --------------------------------------------------------------------------- modelos
def _keras():
    from tensorflow import keras
    return keras, keras.layers


def _compilar(m):
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def _bloques_conv(x, L):
    for f in (16, 32, 64):
        x = L.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.MaxPool2D(2)(x)
    return x


def build_cnn(shape, n):
    keras, L = _keras()
    inp = L.Input(shape)
    x = _bloques_conv(inp, L)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    return _compilar(keras.Model(inp, L.Dense(n, activation="softmax")(x)))


def build_lstm(shape, n):
    keras, L = _keras()
    inp = L.Input(shape)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(inp)
    x = L.Dropout(0.3)(x)
    x = L.Bidirectional(L.LSTM(32))(x)
    x = L.Dropout(0.3)(x)
    return _compilar(keras.Model(inp, L.Dense(n, activation="softmax")(x)))


def build_cnn_lstm(shape, n):
    keras, L = _keras()
    inp = L.Input(shape)
    x = _bloques_conv(inp, L)                                  # (freq, tiempo, canales)
    x = L.Permute((2, 1, 3))(x)                                # (tiempo, freq, canales)
    x = L.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
    w = L.Softmax(axis=1)(L.Dense(1, activation="tanh")(x))    # atención temporal
    x = L.GlobalAveragePooling1D()(L.Multiply()([x, w]))
    x = L.Dropout(0.3)(x)
    return _compilar(keras.Model(inp, L.Dense(n, activation="softmax")(x)))


ARQ = {"cnn": ("mel", build_cnn), "lstm": ("seq", build_lstm), "cnn_lstm": ("mel", build_cnn_lstm)}


# --------------------------------------------------------------------------- entrenamiento
def evaluar(nombre, model, X, y, clases):
    pred = model.predict(X, verbose=0).argmax(1)
    idx = list(range(len(clases)))
    print(f"\n=== {nombre} (hablantes no vistos) ===")
    print(classification_report(y, pred, labels=idx, target_names=clases, zero_division=0))
    rep = classification_report(y, pred, labels=idx, target_names=clases,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=idx)
    ConfusionMatrixDisplay(cm, display_labels=clases).plot(cmap="Blues", xticks_rotation=30)
    plt.title(nombre); plt.tight_layout()
    plt.savefig(f"matriz_confusion_{nombre}.png", dpi=150); plt.close()
    return {"modelo": nombre, "exactitud": rep["accuracy"],
            "precision_macro": rep["macro avg"]["precision"],
            "recall_macro": rep["macro avg"]["recall"],
            "f1_macro": rep["macro avg"]["f1-score"], "parametros": model.count_params()}


def train(a):
    keras, _ = _keras()
    keras.utils.set_random_seed(a.seed)
    mel, seq, labels, spk = construir_dataset(a.data, a.cache, a.recomputar)
    clases = sorted(set(labels))
    y = np.array([clases.index(l) for l in labels])
    tr, va, te = split_by_speaker(spk, seed=a.seed)
    print(f"{len(y)} audios | {len(set(spk))} hablantes | train={len(tr)} val={len(va)} test={len(te)}")
    print("Distribución:", dict(zip(*np.unique(labels, return_counts=True))))

    faltantes = sorted(set(y) - set(y[tr]))
    if faltantes:
        nombres = [clases[i] for i in faltantes]
        raise ValueError(
            "El conjunto train no contiene ejemplos de todas las clases "
            f"({', '.join(nombres)}). Añade hablantes o ajusta los datos."
        )

    mu = seq[tr].reshape(-1, N_FEAT).mean(0)                   # estadísticos solo del train
    sd = seq[tr].reshape(-1, N_FEAT).std(0) + 1e-6
    seq_n = (seq - mu) / sd
    np.savez("norm_seq.npz", mu=mu, sd=sd)
    json.dump(clases, open("clases_prosodia.json", "w"))

    pesos = compute_class_weight("balanced", classes=np.arange(len(clases)), y=y[tr])
    cw = dict(enumerate(pesos))
    datos = {"mel": mel, "seq": seq_n}

    resultados = []
    for nombre in (ARQ if a.arch == "all" else [a.arch]):
        entrada, builder = ARQ[nombre]
        X = datos[entrada]
        model = builder(X.shape[1:], len(clases))
        print(f"Entrenando {nombre}")
        historial = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=a.epochs, batch_size=32,
                  class_weight=cw, verbose=2,
                  callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
        pd.DataFrame(historial.history).rename_axis("epoca").set_axis(
            range(1, len(historial.epoch) + 1)
        ).to_csv(f"historial_{nombre}.csv")
        resultados.append(evaluar(nombre, model, X[te], y[te], clases))
        destino = Path(f"modelo_prosodia_{nombre}.keras")
        temporal = destino.with_name(f".{destino.stem}.pendiente.keras")
        model.save(temporal)
        temporal.replace(destino)

    tabla = pd.DataFrame(resultados).round(3)
    print("\n=== Comparación ===\n", tabla.to_string(index=False))
    tabla.to_csv("comparacion_modelos.csv", index=False)


def predict(a):
    keras, _ = _keras()
    model = keras.models.load_model(f"modelo_prosodia_{a.arch}.keras")
    clases = json.load(open("clases_prosodia.json"))
    y = load_audio(a.file, dur=DUR)
    if ARQ[a.arch][0] == "mel":
        x = mel_fijo(y)[None]
    else:
        n = np.load("norm_seq.npz")
        x = ((secuencia(y) - n["mu"]) / n["sd"])[None]
    p = model.predict(x, verbose=0)[0]
    for c, v in sorted(zip(clases, p), key=lambda t: -t[1]):
        print(f"{c:22s} {v:.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/prosodia")
    t.add_argument("--arch", choices=["all", *ARQ], default="all")
    t.add_argument("--cache", default="cache_prosodia.npz")
    t.add_argument("--recomputar", action="store_true")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("predict")
    p.add_argument("--file", required=True)
    p.add_argument("--arch", choices=list(ARQ), default="cnn_lstm")
    args = ap.parse_args()
    {"train": train, "predict": predict}[args.cmd](args)
