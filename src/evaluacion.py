
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ==========================================
# 1. METRICAS DE EVALUACION
# ==========================================

def calcular_metricas(y_real, y_pred):
    """
    Calcula MAE, MSE y RMSE.
    """

    mae = mean_absolute_error(y_real, y_pred)

    mse = mean_squared_error(y_real, y_pred)

    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }


# ==========================================
# 2. MODELO INGENUO
# ==========================================

def evaluar_modelo_ingenuo(
    resultados,
    scaler_X,
    scaler_y,
    conjunto="val"
):
    """
    Predice que el precio futuro del Brent será
    igual al último precio conocido.

    Devuelve las métricas en dólares por barril.
    """

    filas = []

    # El Brent ocupa la primera columna de X
    indice_brent = 0

    for horizonte, datos in resultados.items():

        X = datos[f"X_{conjunto}"]
        y = datos[f"y_{conjunto}"]

        # Último Brent de cada ventana normalizada
        brent_ultimo = X[:, -1, indice_brent]

        # Recuperar precio original con scaler_X
        y_pred = (
            brent_ultimo * scaler_X.scale_[indice_brent]
            + scaler_X.mean_[indice_brent]
        )

        # Recuperar valores reales en dólares
        y_real = scaler_y.inverse_transform(
            y.reshape(-1, 1)
        ).ravel()

        metricas = calcular_metricas(y_real, y_pred)

        filas.append({
            "Horizonte": horizonte,
            "Modelo": "Ingenuo",
            **metricas,
        })

    return pd.DataFrame(filas)
