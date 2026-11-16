import json
import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import torch

from src.preprocessing import preparar_datasets
from src.modelos import crear_modelo, contar_parametros
from src.entrenamiento import (
    fijar_semilla,
    entrenar_modelo,
    predecir_modelo
)
from src.evaluacion import (
    calcular_metricas,
    evaluar_modelo_ingenuo
)


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "petroleo_dataset.csv"

FECHA = datetime.now(
    ZoneInfo("Europe/Madrid")
).strftime("%d_%m_%Y")

CARPETA_RESULTADOS = (
    RAIZ / "resultados" / "reentrenamiento" / FECHA
)

MODELOS = ["RNN", "LSTM", "GRU"]
HORIZONTES = [5, 63, 252]

VENTANA = 60
EPOCAS = 30
BATCH_SIZE = 32
LR = 0.001
PACIENCIA = 5
SEMILLA = 42


# ==========================================
# 2. REENTRENAMIENTO
# ==========================================

def reentrenar_modelos():

    CARPETA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    datos = pd.read_csv(
        RUTA_DATOS,
        parse_dates=["Date"],
        index_col="Date"
    ).sort_index()

    print("Última fecha:", datos.index.max())
    print("Dimensiones:", datos.shape)

    preprocesado = preparar_datasets(
        datos,
        ventana=VENTANA,
        horizontes=HORIZONTES
    )

    resultados = preprocesado["resultados"]
    scaler_X = preprocesado["scaler_X"]
    scaler_y = preprocesado["scaler_y"]

    variables = preprocesado["variables"]

    metricas_totales = []
    historiales = {}

    # Guardar escaladores
    joblib.dump(
        scaler_X,
        CARPETA_RESULTADOS / "scaler_X.pkl"
    )

    joblib.dump(
        scaler_y,
        CARPETA_RESULTADOS / "scaler_y.pkl"
    )

    # Modelo ingenuo
    metricas_ingenuo = evaluar_modelo_ingenuo(
        resultados=resultados,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        conjunto="val"
    )

    metricas_totales.extend(
        metricas_ingenuo.to_dict("records")
    )

    for horizonte in HORIZONTES:

        print("\n" + "=" * 60)
        print(f"HORIZONTE: {horizonte} DÍAS")
        print("=" * 60)

        conjuntos = resultados[horizonte]

        X_train = conjuntos["X_train"]
        y_train = conjuntos["y_train"]

        X_val = conjuntos["X_val"]
        y_val = conjuntos["y_val"]

        y_real = scaler_y.inverse_transform(
            y_val.reshape(-1, 1)
        ).ravel()

        for tipo in MODELOS:

            fijar_semilla(SEMILLA)

            modelo = crear_modelo(
                tipo=tipo,
                input_size=len(variables),
                hidden_size_1=64,
                hidden_size_2=32,
                dropout=0.2,
                residual=False
            )

            modelo, historial = entrenar_modelo(
                modelo=modelo,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epocas=EPOCAS,
                batch_size=BATCH_SIZE,
                lr=LR,
                paciencia=PACIENCIA,
                semilla=SEMILLA
            )

            nombre = f"{tipo.lower()}_h{horizonte}"

            # Predicciones
            pred_norm = predecir_modelo(
                modelo,
                X_val
            )

            y_pred = scaler_y.inverse_transform(
                pred_norm.reshape(-1, 1)
            ).ravel()

            metricas = calcular_metricas(
                y_real,
                y_pred
            )

            metricas_totales.append({
                "Modelo": tipo,
                "Horizonte": horizonte,
                **metricas,
                "Parámetros": contar_parametros(modelo),
                "Tiempo (s)": historial["tiempo_segundos"]
            })

            historiales[nombre] = historial

            # Guardar pesos compatibles con CPU
            pesos = {
                clave: valor.detach().cpu()
                for clave, valor in modelo.state_dict().items()
            }

            torch.save(
                pesos,
                CARPETA_RESULTADOS / f"{nombre}.pth"
            )

    # ==========================================
    # 3. GUARDAR RESULTADOS
    # ==========================================

    tabla = pd.DataFrame(metricas_totales)

    tabla.to_csv(
        CARPETA_RESULTADOS / "metricas.csv",
        index=False
    )

    with open(
        CARPETA_RESULTADOS / "historiales.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(historiales, f, indent=4)

    configuracion = {
        "fecha_reentrenamiento": FECHA,
        "ultima_fecha_datos": str(datos.index.max().date()),
        "dataset_sha256": hashlib.sha256(
            RUTA_DATOS.read_bytes()
        ).hexdigest(),
        "objetivo": "prediccion_directa_precio_brent",
        "modelos": MODELOS,
        "horizontes": HORIZONTES,
        "ventana": VENTANA,
        "variables": list(variables),
        "numero_variables": len(variables),
        "capas_recurrentes": [64, 32],
        "dropout": 0.2,
        "residual": False,
        "funcion_perdida": "HuberLoss",
        "delta": 0.5,
        "epocas": EPOCAS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "paciencia": PACIENCIA,
        "semilla": SEMILLA
    }

    with open(
        CARPETA_RESULTADOS / "configuracion.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(configuracion, f, indent=4)

    print("\nREENTRENAMIENTO FINALIZADO")
    print(tabla.round(3))
    print("\nResultados guardados en:")
    print(CARPETA_RESULTADOS)


if __name__ == "__main__":
    reentrenar_modelos()