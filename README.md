# Reconocimiento de voz con aprendizaje profundo

Clasificación de voz por prosodia con tres arquitecturas: CNN sobre espectrogramas Mel, BiLSTM sobre descriptores acústicos y CNN-BiLSTM con atención temporal.

## Datos

El script recorre las carpetas de forma recursiva. La primera carpeta dentro de la ruta indicada se toma como clase y el identificador del hablante se obtiene del inicio del nombre del WAV (antes del primer guion bajo).

```text
speech/
├── normal/
│   └── f1/f1_n_01.wav
└── shout/
    └── f1/f1_s_01.wav
```

El dataset RISC del proyecto está organizado así y ofrece las clases `normal` y `shout`. La separación de entrenamiento, validación y prueba se hace por hablante para evitar que la misma voz aparezca en más de una partición. Se requieren al menos tres hablantes distintos.

## Instalación

Usa Python 3.10, 3.11 o 3.12 y crea un entorno virtual. Instala las dependencias con:

```bash
python -m pip install -r requirements.txt
```

TensorFlow puede requerir una instalación específica según el sistema operativo y el hardware. Consulta sus instrucciones oficiales si la instalación indicada en `requirements.txt` no está disponible para tu plataforma.

## Uso

Entrenar las tres arquitecturas:

```bash
python prosodia_dl.py train --data data/prosodia --arch all
```

Para usar el dataset RISC que está junto a este proyecto:

```bash
python prosodia_dl.py train --data ..\RISC\RISC\speech --arch all
```

Entrenar solo una arquitectura:

```bash
python prosodia_dl.py train --data data/prosodia --arch cnn_lstm --epochs 60
```

Clasificar un audio usando un modelo entrenado:

```bash
python prosodia_dl.py predict --file audio.wav --arch cnn_lstm
```

El entrenamiento guarda los modelos `.keras`, la normalización de secuencias, las clases, una tabla CSV de resultados y matrices de confusión en el directorio actual. `cache_prosodia.npz` evita recalcular las características acústicas; usa `--recomputar` para regenerarla.

## Interfaz local

Después de instalar las dependencias, ejecuta `python interfaz.py` desde la
carpeta del proyecto. La interfaz usa Tkinter, incluido en la instalación
estándar de Python para Windows.

- Elige la carpeta de clases (por ejemplo, `..\RISC\RISC\speech`) o un ZIP
  con esa misma estructura. El ZIP puede tener carpetas contenedoras por encima
  de las clases. Mantén el identificador del hablante al inicio de cada WAV.
- Selecciona la arquitectura, las épocas y la semilla. El registro muestra el
  procesamiento de audio y las métricas por época.
- En Resultados y predicción puedes seleccionar una corrida, abrir sus curvas
  y matriz de confusión, exportar el CSV y clasificar un WAV.

Cada corrida queda en `corridas/` con su configuración, resumen del dataset,
registro, caché y resultados. Si se usa un ZIP, se guardan ahí sus WAV extraídos.
Los resultados anteriores se pueden abrir desde la carpeta del proyecto o su
carpeta superior. Las curvas están disponibles para las corridas nuevas.
