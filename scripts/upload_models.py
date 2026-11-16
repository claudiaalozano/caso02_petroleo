import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from huggingface_hub import HfApi


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RAIZ = Path(__file__).resolve().parent.parent

FECHA = datetime.now(
    ZoneInfo("Europe/Madrid")
).strftime("%d_%m_%Y")

CARPETA_MODELOS = (
    RAIZ / "resultados" / "reentrenamiento" / FECHA
)

REPO_ID = os.environ["HF_REPO_ID"]


# ==========================================
# 2. SUBIDA DE MODELOS
# ==========================================

def subir_modelos():

    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError(
            "No se ha encontrado HF_TOKEN"
        )

    if not CARPETA_MODELOS.is_dir():
        raise FileNotFoundError(
            f"No existe {CARPETA_MODELOS}"
        )

    archivos_esperados = [
        f"{modelo}_h{horizonte}.pth"
        for modelo in ["rnn", "lstm", "gru"]
        for horizonte in [5, 63, 252]
    ] + [
        "scaler_X.pkl",
        "scaler_y.pkl",
        "metricas.csv",
        "historiales.json",
        "configuracion.json"
    ]

    for archivo in archivos_esperados:

        if not (CARPETA_MODELOS / archivo).is_file():
            raise FileNotFoundError(
                f"Falta el archivo {archivo}"
            )

    api = HfApi(token=token)

    print("Subiendo modelos a Hugging Face...")
    print("Repositorio:", REPO_ID)
    print("Fecha:", FECHA)

    api.upload_folder(
        folder_path=str(CARPETA_MODELOS),
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo=FECHA,
        commit_message=f"Retrained oil models - {FECHA}"
    )

    print("Modelos subidos correctamente.")
    print(f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    subir_modelos()