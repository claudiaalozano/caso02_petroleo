import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "resultados" / "test_final"
SALIDA = RAIZ / "app" / "public" / "datos" / "evaluacion.json"

HORIZONTES = [5, 63, 252]
OBJETIVOS = ["retorno", "nivel"]
MODELOS = ["RNN", "LSTM", "GRU"]

resultados = {}

for objetivo in OBJETIVOS:
    for h in HORIZONTES:

        ruta = CARPETA / f"predicciones_{objetivo}_h{h}.csv"

        if not ruta.exists():
            raise FileNotFoundError(ruta)

        df = pd.read_csv(ruta)

        requeridas = [
            "Fecha", "Real", "Ingenuo", *MODELOS
        ]

        faltantes = [
            c for c in requeridas if c not in df.columns
        ]

        if faltantes:
            raise ValueError(
                f"Faltan columnas en {ruta.name}: {faltantes}"
            )

        df["Fecha"] = pd.to_datetime(df["Fecha"])

        for c in requeridas[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.sort_values("Fecha")

        clave = f"{objetivo}_{h}"
        resultados[clave] = {}

        for modelo in MODELOS:

            sub = df[
                ["Fecha", "Real", "Ingenuo", modelo]
            ].dropna().copy()

            if sub.empty:
                raise ValueError(
                    f"Sin predicciones: {objetivo}, {h}, {modelo}"
                )

            real = sub["Real"].to_numpy(dtype=float)
            pred = sub[modelo].to_numpy(dtype=float)
            origen = sub["Ingenuo"].to_numpy(dtype=float)

            errores = real - pred

            mae = mean_absolute_error(real, pred)
            rmse = np.sqrt(mean_squared_error(real, pred))
            r2 = r2_score(real, pred)

            mae_ingenuo = mean_absolute_error(real, origen)
            skill = (
                1 - mae / mae_ingenuo
                if mae_ingenuo > 0
                else None
            )

            cambio_real = real - origen
            cambio_pred = pred - origen

            # Clasificación secundaria: sube frente a no sube.
            clases = (cambio_real > 0).astype(int)

            # Puntuación continua: cambio porcentual predicho.
            # No se utilizan etiquetas binarias como score.
            score = np.divide(
                cambio_pred,
                origen,
                out=np.zeros_like(cambio_pred),
                where=origen != 0,
            )

            auc = None
            puntos_roc = []

            if len(np.unique(clases)) == 2:
                fpr, tpr, umbrales = roc_curve(clases, score)
                auc = float(roc_auc_score(clases, score))

                puntos_roc = [
                    {
                        "fpr": float(x),
                        "tpr": float(y),
                    }
                    for x, y in zip(fpr, tpr)
                ]

            acierto = float(
                np.mean((cambio_pred > 0) == (cambio_real > 0))
                * 100
            )

            # Ajuste lineal descriptivo para el diagrama de puntos.
            # NO representa un nuevo modelo predictivo.
            if np.std(real) > 0:
                pendiente, intercepto = np.polyfit(real, pred, 1)
            else:
                pendiente, intercepto = 0.0, float(np.mean(pred))

            puntos = [
                {
                    "fecha": fecha.strftime("%Y-%m-%d"),
                    "real": float(y),
                    "predicho": float(p),
                    "ingenuo": float(n),
                    "residuo": float(y - p),
                }
                for fecha, y, p, n in zip(
                    sub["Fecha"], real, pred, origen
                )
            ]

            resultados[clave][modelo] = {
                "objetivo": objetivo,
                "horizonte": h,
                "modelo": modelo,
                "n": len(sub),
                "metricas": {
                    "MAE": float(mae),
                    "RMSE": float(rmse),
                    "R2": float(r2),
                    "Skill": (
                        float(skill) if skill is not None else None
                    ),
                    "Sesgo": float(np.mean(pred - real)),
                    "Acierto_direccional": acierto,
                    "AUC_direccion": auc,
                },
                "regresion": {
                    "pendiente": float(pendiente),
                    "intercepto": float(intercepto),
                },
                "roc": puntos_roc,
                "puntos": puntos,
            }

            print(
                f"{modelo:5} | {objetivo:7} | "
                f"h={h:3} | n={len(sub):4} | "
                f"MAE={mae:.3f} | "
                f"AUC={auc if auc is not None else 'N/D'}"
            )

SALIDA.parent.mkdir(parents=True, exist_ok=True)

SALIDA.write_text(
    json.dumps(
        resultados,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ),
    encoding="utf-8",
)

print("\nEXPORTACIÓN FINALIZADA")
print("Archivo:", SALIDA)
print("Combinaciones:", sum(len(v) for v in resultados.values()))
