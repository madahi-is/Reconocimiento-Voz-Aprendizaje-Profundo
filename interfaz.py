"""Interfaz local para entrenar y consultar modelos de prosodia."""

import csv
from collections import Counter
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
from importlib.util import find_spec
from datetime import datetime
from pathlib import Path, PurePosixPath
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parent
RUNS = PROJECT / "corridas"
ARCH = ("cnn", "lstm", "cnn_lstm")
PERCENT_METRICS = {"exactitud", "precision_macro", "recall_macro", "f1_macro"}


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))
TRAIN_MODULES = ("pandas", "numpy", "librosa", "matplotlib", "sklearn", "tensorflow")


def dependency_error(modules):
    missing = [name for name in modules if find_spec(name) is None]
    if missing:
        return (
            f"Faltan módulos: {', '.join(missing)}.\n"
            f"Intérprete: {sys.executable}\n\n"
            "Para instalar las dependencias con este Python, ejecuta en PowerShell:\n"
            f'& "{sys.executable}" -m pip install -r "{PROJECT / "requirements.txt"}"'
        )
    return None


def relaunch_with_installed_python():
    """Usa la instalación local si el intérprete del editor no tiene los paquetes."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data or not dependency_error(TRAIN_MODULES):
        return False
    python = Path(local_app_data) / "Programs" / "Python" / "Python312" / "python.exe"
    if not python.is_file() or python.resolve() == Path(sys.executable).resolve():
        return False
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    subprocess.Popen([str(python), str(Path(__file__).resolve())], env=env)
    return True


def unpack_wavs(source, target):
    with ZipFile(source) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename.replace(chr(92), "/"))
            if entry.is_dir() or path.suffix.lower() != ".wav":
                continue
            if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
                raise ValueError(f"Ruta inválida en el ZIP: {entry.filename}")
            destination = (target / path).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError(f"Ruta inválida en el ZIP: {entry.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as stream, destination.open("xb") as output:
                shutil.copyfileobj(stream, output)
    while True:
        children = list(target.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            return target
        target = children[0]


def prepare_dataset(source, run):
    import pandas as pd
    from audio_utils import split_by_speaker

    root = Path(source)
    if root.is_file():
        target = run / "dataset"
        target.mkdir()
        root = unpack_wavs(root, target)
    if not root.is_dir():
        raise ValueError("Selecciona una carpeta de clases o un ZIP con audios WAV.")
    rows = []
    for file in sorted(root.rglob("*.wav")):
        parts = file.relative_to(root).parts
        if len(parts) < 2:
            raise ValueError("Coloca cada WAV dentro de una carpeta de clase.")
        rows.append((parts[0], file.stem.split("_")[0]))
    data = pd.DataFrame(rows, columns=["clase", "hablante"])
    if data.clase.nunique() < 2:
        raise ValueError("Se necesitan al menos dos carpetas de clases con WAV.")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for name, indices in zip(("train", "val", "test"), split_by_speaker(data.hablante, seed=config["seed"])):
        data.loc[indices, "particion"] = name
    summary = data.groupby(["clase", "hablante", "particion"]).size()
    summary.rename("audios").reset_index().to_csv(run / "dataset.csv", index=False)
    print(data.groupby(["clase", "particion"]).size().unstack(fill_value=0).to_string())
    return root


def worker(run):
    error = dependency_error(TRAIN_MODULES)
    if error:
        print(error, file=sys.stderr)
        return 1
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    print("Preparando dataset")
    root = prepare_dataset(config["dataset"], run)
    script = PROJECT / "prosodia_dl.py"
    sys.argv = [
        str(script), "train",
        "--data", str(root), "--arch", config["arch"], "--epochs", str(config["epochs"]),
        "--seed", str(config["seed"]),
    ]
    runpy.run_path(str(script), run_name="__main__")
    return 0


class App:
    def __init__(self, window):
        self.window = window
        self.process = None
        self.log_path = None
        self.log_offset = 0
        self.prediction_dir = None
        self.training_run = None
        self.stop_requested = False
        window.title("Laboratorio de prosodia")
        window.geometry("960x680")
        tabs = ttk.Notebook(window)
        tabs.pack(fill="both", expand=True, padx=12, pady=12)
        train = ttk.Frame(tabs, padding=12)
        results = ttk.Frame(tabs, padding=12)
        tabs.add(train, text="Dataset y entrenamiento")
        tabs.add(results, text="Resultados y predicción")

        risc = PROJECT.parent / "RISC" / "RISC" / "speech"
        self.dataset = tk.StringVar(value=str(risc) if risc.is_dir() else "")
        self.arch = tk.StringVar(value="cnn")
        self.epochs = tk.StringVar(value="10")
        self.seed = tk.StringVar(value="42")
        ttk.Label(train, text="Carpeta de clases o ZIP con audios WAV").pack(anchor="w")
        ttk.Entry(train, textvariable=self.dataset).pack(fill="x", pady=6)
        actions = ttk.Frame(train)
        actions.pack(fill="x")
        ttk.Button(actions, text="Elegir carpeta", command=lambda: self.choose(False)).pack(side="left")
        ttk.Button(actions, text="Elegir ZIP", command=lambda: self.choose(True)).pack(side="left")
        settings = ttk.Frame(train)
        settings.pack(fill="x", pady=10)
        ttk.Label(settings, text="Arquitectura").pack(side="left")
        ttk.Combobox(settings, values=(*ARCH, "all"), state="readonly", textvariable=self.arch, width=12).pack(side="left", padx=6)
        for label, variable in (("Épocas", self.epochs), ("Semilla", self.seed)):
            ttk.Label(settings, text=label).pack(side="left")
            ttk.Entry(settings, textvariable=variable, width=8).pack(side="left", padx=6)
        controls = ttk.Frame(train)
        controls.pack(fill="x")
        ttk.Button(controls, text="Entrenar", command=self.train).pack(side="left")
        self.stop_button = ttk.Button(
            controls, text="Detener entrenamiento", command=self.stop_training, state="disabled"
        )
        self.stop_button.pack(side="left", padx=8)
        self.status = tk.StringVar(value="Selecciona el dataset para empezar.")
        ttk.Label(train, textvariable=self.status).pack(anchor="w", pady=6)
        self.log = self.text_area(train)

        filters = ttk.Frame(results)
        filters.pack(fill="x", pady=(0, 10))
        ttk.Label(filters, text="Corrida").pack(side="left")
        self.selection = ttk.Combobox(filters, state="readonly", width=44)
        self.selection.pack(side="left", padx=(6, 18))
        self.selection.bind("<<ComboboxSelected>>", lambda _: self.show_results())
        ttk.Label(filters, text="Arquitectura").pack(side="left")
        self.model = ttk.Combobox(filters, state="readonly", values=ARCH, width=12)
        self.model.set("cnn")
        self.model.pack(side="left", padx=6)
        toolbar = ttk.Frame(results)
        toolbar.pack(fill="x")
        for label, callback in (
            ("Comparar modelos", self.comparison_chart),
            ("Matriz de confusión", self.matrix), ("Curvas", self.curves),
            ("Guardar CSV", self.export), ("Clasificar WAV", self.predict),
        ):
            ttk.Button(toolbar, text=label, command=callback).pack(side="left", padx=(0, 6))
        self.result_tabs = ttk.Notebook(results)
        self.result_tabs.pack(fill="both", expand=True, pady=(12, 0))
        summary = ttk.Frame(self.result_tabs, padding=12)
        dataset = ttk.Frame(self.result_tabs, padding=12)
        self.prediction_tab = ttk.Frame(self.result_tabs, padding=12)
        self.result_tabs.add(summary, text="Resumen")
        self.result_tabs.add(dataset, text="Dataset")
        self.result_tabs.add(self.prediction_tab, text="Predicción")

        self.run_info = ttk.Label(summary, wraplength=760, justify="left")
        self.run_info.pack(anchor="w", pady=(0, 14))
        ttk.Label(summary, text="Métricas en el conjunto de prueba").pack(anchor="w", pady=(0, 6))
        self.metrics_table = self.table(summary, (
            ("modelo", "Modelo"), ("exactitud", "Exactitud"),
            ("precision_macro", "Precisión macro"), ("recall_macro", "Recall macro"),
            ("f1_macro", "F1 macro"), ("parametros", "Parámetros"),
        ), height=4)
        self.dataset_info = ttk.Label(dataset, wraplength=760, justify="left")
        self.dataset_info.pack(anchor="w", pady=(0, 10))
        self.dataset_table = self.table(dataset, (
            ("clase", "Clase"), ("hablante", "Hablante"),
            ("particion", "Partición"), ("audios", "Audios"),
        ), height=12)
        ttk.Label(self.prediction_tab, text="Pulsa «Clasificar WAV» para usar el modelo guardado.").pack(anchor="w")
        self.results = self.text_area(self.prediction_tab)
        self.refresh()
        window.protocol("WM_DELETE_WINDOW", self.close)

    @staticmethod
    def table(parent, columns, height):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=[key for key, _ in columns], show="headings", height=height)
        for key, title in columns:
            tree.heading(key, text=title)
            tree.column(key, width=112, minwidth=80, anchor="center")
        scroll = ttk.Scrollbar(frame, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        tree.tag_configure("alternate", background="#eef4fb")
        return tree

    @staticmethod
    def fill_table(tree, rows):
        tree.delete(*tree.get_children())
        for index, row in enumerate(rows):
            tree.insert("", "end", values=row, tags=("alternate",) if index % 2 else ())

    @staticmethod
    def text_area(parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, pady=6)
        area = tk.Text(frame, wrap="none", state="disabled")
        scroll = ttk.Scrollbar(frame, command=area.yview)
        area.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        area.pack(side="left", fill="both", expand=True)
        return area

    @staticmethod
    def write(area, value, clear=False):
        area.configure(state="normal")
        if clear:
            area.delete("1.0", "end")
        area.insert("end", value)
        area.see("end")
        area.configure(state="disabled")

    def choose(self, zip_file):
        value = filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")]) if zip_file else filedialog.askdirectory()
        if value:
            self.dataset.set(value)

    def start(self, args, cwd, log, area, finished):
        self.log_offset = 0
        self.log_path = log
        with log.open("w", encoding="utf-8") as output:
            self.process = subprocess.Popen(
                [sys.executable, "-u", *args], cwd=cwd, stdout=output,
                stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self.write(area, "", clear=True)
        self.window.after(400, self.poll, area, finished)

    def poll(self, area, finished):
        code = self.process.poll()
        with self.log_path.open(encoding="utf-8", errors="replace") as log:
            log.seek(self.log_offset)
            self.write(area, log.read())
            self.log_offset = log.tell()
        if code is None:
            self.window.after(400, self.poll, area, finished)
        else:
            self.process = None
            finished(code)

    def train(self):
        if self.process:
            messagebox.showinfo("Proceso activo", "Espera a que finalice el proceso actual.")
            return
        error = dependency_error(TRAIN_MODULES)
        if error:
            messagebox.showerror("Dependencias del entrenamiento", error)
            return
        try:
            source = Path(self.dataset.get()).expanduser().resolve(strict=True)
            epochs, seed = int(self.epochs.get()), int(self.seed.get())
            if not self.dataset.get() or epochs < 1 or not 0 <= seed < 2**32:
                raise ValueError("Indica un dataset, épocas positivas y una semilla entre 0 y 4294967295.")
            run = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps({
                "dataset": str(source), "arch": self.arch.get(), "epochs": epochs, "seed": seed,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.start([str(Path(__file__).resolve()), "--worker", str(run)], run, run / "entrenamiento.log", self.log, self.training_finished)
            self.training_run = run
            self.stop_requested = False
            self.stop_button.configure(state="normal")
            self.status.set(f"Entrenando en {run.name}")
        except (ValueError, OSError) as error:
            messagebox.showerror("Entrenamiento", str(error))

    def stop_training(self):
        if self.training_run is None or self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.terminate()
        except OSError as error:
            if self.process.poll() is None:
                messagebox.showerror("Detener entrenamiento", str(error))
            return
        self.stop_requested = True
        self.stop_button.configure(state="disabled")
        self.status.set("Deteniendo entrenamiento…")

    def training_finished(self, code):
        self.stop_button.configure(state="disabled")
        if self.stop_requested and code != 0:
            message = "Entrenamiento detenido por el usuario."
            with (self.training_run / "entrenamiento.log").open("a", encoding="utf-8") as log:
                log.write(f"\n{message}\n")
            self.write(self.log, f"\n{message}\n")
            (self.training_run / "detenida").touch()
        else:
            message = "Entrenamiento terminado" if code == 0 else f"El entrenamiento falló ({code}). Revisa el registro."
        self.status.set(message)
        self.training_run = None
        self.refresh()

    def refresh(self):
        previous = self.selection.get()
        self.folders = {p.name: p for p in sorted(RUNS.glob("*/"), reverse=True)}
        for path, label in ((PROJECT.parent, "Resultados previos (carpeta superior)"), (PROJECT, "Resultados previos (proyecto)")):
            if (path / "comparacion_modelos.csv").exists():
                self.folders[label] = path
        self.selection.configure(values=list(self.folders))
        self.selection.set(previous if previous in self.folders else next(iter(self.folders), ""))
        self.show_results()

    def selected(self):
        return self.folders.get(self.selection.get())

    def show_results(self):
        folder = self.selected()
        self.fill_table(self.metrics_table, [])
        self.fill_table(self.dataset_table, [])
        self.dataset_info.configure(text="El detalle del dataset no está disponible para esta corrida.")
        if folder is None:
            self.run_info.configure(text="Todavía no hay corridas.")
            return
        config_file = folder / "config.json"
        lines = []
        if (folder / "detenida").is_file():
            lines.append("Estado   Detenida por el usuario")
        if config_file.is_file():
            config = json.loads(config_file.read_text(encoding="utf-8-sig"))
            lines.append(f"Dataset   {config.get('dataset', '—')}")
            lines.append(f"Arquitectura   {str(config.get('arch', '—')).upper()}     Épocas configuradas   {config.get('epochs', '—')}     Semilla   {config.get('seed', '—')}")
        metrics = read_rows(folder / "comparacion_modelos.csv")
        available = [arch for arch in ARCH if (folder / f"modelo_prosodia_{arch}.keras").is_file()]
        self.model.configure(values=available or ARCH)
        if self.model.get() not in (available or ARCH):
            self.model.set((available or ARCH)[0])
        lines.append("Modelos guardados   " + (", ".join(arch.upper() for arch in available) or "Ninguno disponible"))
        if not metrics:
            lines.append("Todavía no hay métricas de prueba para esta corrida.")
        self.run_info.configure(text="\n".join(lines))
        self.fill_table(self.metrics_table, [
            [
                f"{float(row[key]) * 100:.2f} %" if key in PERCENT_METRICS and row.get(key)
                else row.get(key, "—")
                for key in self.metrics_table["columns"]
            ]
            for row in metrics
        ])

        data = read_rows(folder / "dataset.csv")
        if data:
            counts = Counter()
            speakers = {row["hablante"] for row in data}
            partitions = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
            for row in data:
                counts[row["particion"]] += int(row["audios"])
            totals = "     ".join(f"{label}   {counts[key]:,}" for key, label in partitions.items())
            self.dataset_info.configure(text=f"{sum(counts.values()):,} audios · {len(speakers)} hablantes\n{totals}")
            self.fill_table(self.dataset_table, [
                (row["clase"], row["hablante"], partitions.get(row["particion"], row["particion"]), row["audios"])
                for row in data
            ])

    def artifact(self, name):
        folder = self.selected()
        path = folder / name if folder else None
        if path and path.is_file():
            return path
        messagebox.showinfo("Archivo no disponible", f"La corrida seleccionada no contiene {name}.")
        return None

    def matrix(self):
        path = self.artifact(f"matriz_confusion_{self.model.get()}.png")
        if path:
            popup = tk.Toplevel(self.window)
            popup.title("Matriz de confusión")
            picture = tk.PhotoImage(file=str(path)).subsample(2)
            label = ttk.Label(popup, image=picture)
            label.picture = picture
            label.pack()

    def comparison_chart(self):
        path = self.artifact("comparacion_modelos.csv")
        if not path:
            return
        import pandas as pd
        from matplotlib.figure import Figure
        from matplotlib.ticker import PercentFormatter

        metrics = {
            "exactitud": "Exactitud",
            "precision_macro": "Precisión macro",
            "recall_macro": "Recall macro",
            "f1_macro": "F1 macro",
        }
        results = pd.read_csv(path).set_index("modelo")
        figure = Figure(figsize=(10, 5), layout="tight")
        axis = figure.add_subplot()
        results[list(metrics)].rename(columns=metrics).plot.bar(ax=axis, rot=0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.set(ylim=(0, 1.16), yticks=[0, .2, .4, .6, .8, 1],
                 xlabel="Modelo", ylabel="Puntaje", title="Resultados en el conjunto de prueba")
        axis.legend(loc="upper center", ncol=4, frameon=False)
        for bars in axis.containers:
            axis.bar_label(bars, labels=[f"{value:.1%}" for value in bars.datavalues], padding=2, fontsize=8)
        self.show_figure("Comparación de modelos", figure)

    def curves(self):
        path = self.artifact(f"historial_{self.model.get()}.csv")
        if not path:
            return
        import pandas as pd
        from matplotlib.figure import Figure
        from matplotlib.ticker import PercentFormatter

        history = pd.read_csv(path).set_index("epoca")
        figure = Figure(figsize=(9, 4), layout="tight")
        for ax, metric, label in zip(figure.subplots(1, 2), ("loss", "accuracy"), ("Pérdida", "Exactitud")):
            series = history[[metric, f"val_{metric}"]].rename(columns={
                metric: "Entrenamiento", f"val_{metric}": "Validación",
            })
            series.plot(ax=ax, title=label)
            if metric == "accuracy":
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        self.show_figure("Curvas por época", figure)

    def show_figure(self, title, figure):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        popup = tk.Toplevel(self.window)
        popup.title(title)
        canvas = FigureCanvasTkAgg(figure, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def export(self):
        path = self.artifact("comparacion_modelos.csv")
        if path:
            destination = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=path.name)
            if destination and Path(destination).resolve() != path.resolve():
                shutil.copyfile(path, destination)

    def predict(self):
        if self.process:
            messagebox.showinfo("Proceso activo", "Espera a que finalice el proceso actual.")
            return
        arch = self.model.get()
        model = self.artifact(f"modelo_prosodia_{arch}.keras")
        if not model:
            return
        wav = filedialog.askopenfilename(filetypes=[("Audio WAV", "*.wav")])
        if wav:
            self.result_tabs.select(self.prediction_tab)
            self.prediction_dir = tempfile.TemporaryDirectory()
            self.start([str(PROJECT / "prosodia_dl.py"), "predict", "--file", wav, "--arch", arch],
                       model.parent, Path(self.prediction_dir.name) / "prediccion.log", self.results,
                       self.prediction_finished)

    def prediction_finished(self, code):
        if code:
            self.write(self.results, f"\nLa predicción falló ({code}).")
        self.prediction_dir.cleanup()
        self.prediction_dir = None

    def close(self):
        if self.process:
            messagebox.showinfo("Proceso activo", "Espera a que termine el proceso antes de cerrar.")
        else:
            self.window.destroy()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.exit(worker(Path(sys.argv[2])))
    if relaunch_with_installed_python():
        sys.exit(0)
    window = tk.Tk()
    App(window)
    window.mainloop()
