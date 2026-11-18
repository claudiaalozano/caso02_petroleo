
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ==========================================
# 1. METRICAS DE EVALUACION
# ==========================================

def calcular_metricas(y_real, y_pred, precio_origen=None):
    """
    Calcula MAE, MSE, RMSE y el sesgo medio (pred - real).

    Si se indica precio_origen (último precio conocido al predecir),
    añade el acierto direccional: porcentaje de veces que el modelo
    acierta si el precio sube o baja respecto al origen.
    El modelo ingenuo no tiene dirección (predice variación 0).
    """

    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_real, y_pred)

    mse = mean_squared_error(y_real, y_pred)

    rmse = np.sqrt(mse)

    metricas = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Sesgo": float(np.mean(y_pred - y_real)),
    }

    if precio_origen is not None:
        origen = np.asarray(precio_origen, dtype=float)
        dir_pred = np.sign(y_pred - origen)
        dir_real = np.sign(y_real - origen)
        validos = (dir_pred != 0) & (dir_real != 0)
        metricas["Acierto dir. (%)"] = (
            100 * float(np.mean(dir_pred[validos] == dir_real[validos]))
            if validos.any() else np.nan
        )

    return metricas


def skill_vs_ingenuo(mae_modelo, mae_ingenuo):
    """1 - MAE/MAE_ingenuo. Positivo = mejor que el ingenuo."""

    return 1 - mae_modelo / mae_ingenuo


# ==========================================
# 2. CONTRACCION HACIA EL MODELO INGENUO
# ==========================================

def contraer_prediccion(y_pred, precio_origen, alfa):
    """
    Contrae la predicción hacia el último precio conocido:

        precio = origen * exp(alfa * log(pred / origen))

    alfa = 0 -> modelo ingenuo; alfa = 1 -> predicción sin cambios.
    Útil cuando la red tiene algo de señal pero mucho ruido.
    """

    origen = np.asarray(precio_origen, dtype=float)
    retorno = np.log(np.asarray(y_pred, dtype=float) / origen)

    return origen * np.exp(alfa * retorno)


def elegir_alfa(y_real, y_pred, precio_origen, rejilla=np.linspace(0, 1, 21)):
    """
    Elige el alfa de la rejilla que minimiza el MAE.

    Debe llamarse con predicciones FUERA DE MUESTRA de un periodo
    anterior al que se va a evaluar (p. ej. la validación interna).
    """

    maes = [
        mean_absolute_error(y_real, contraer_prediccion(y_pred, precio_origen, a))
        for a in rejilla
    ]

    return float(rejilla[int(np.argmin(maes))])


# ==========================================
# 3. TEST DE DIEBOLD-MARIANO
# ==========================================

def test_diebold_mariano(y_real, pred_modelo, pred_referencia, horizonte=1,
                         potencia=1):
    """
    Contrasta si la precisión de dos predicciones es distinta.

    H0: misma precisión esperada.
    d_t = |e_modelo|^p - |e_referencia|^p   (p=1 -> MAE, p=2 -> MSE)

    Usa varianza HAC (Newey-West) con h-1 retardos, porque los errores
    a h pasos con ventanas solapadas están autocorrelacionados, y la
    corrección de muestra pequeña de Harvey, Leybourne y Newbold (1997).

    Devuelve el estadístico y el p-valor bilateral.
    Estadístico < 0 -> el modelo tiene menos error que la referencia.
    """

    y_real = np.asarray(y_real, dtype=float)
    e1 = np.abs(y_real - np.asarray(pred_modelo, dtype=float)) ** potencia
    e2 = np.abs(y_real - np.asarray(pred_referencia, dtype=float)) ** potencia
    d = e1 - e2

    n = len(d)

    # Con menos observaciones que el horizonte, la varianza HAC no se puede
    # estimar (los errores solapados de una ventana de h días cubren toda la
    # muestra): el contraste no es válido.
    if horizonte >= n:
        return np.nan, np.nan

    d_media = d.mean()
    dc = d - d_media

    varianza = dc @ dc / n
    for k in range(1, horizonte):
        varianza += 2 * (dc[k:] @ dc[:-k]) / n

    if varianza <= 0:
        return np.nan, np.nan

    estadistico = d_media / np.sqrt(varianza / n)

    correccion = np.sqrt(
        (n + 1 - 2 * horizonte + horizonte * (horizonte - 1) / n) / n
    )
    estadistico *= correccion

    p_valor = 2 * stats.t.sf(abs(estadistico), df=n - 1)

    return float(estadistico), float(p_valor)


# ==========================================
# 4. MODELO INGENUO
# ==========================================

def evaluar_modelo_ingenuo(
    resultados,
    scaler_X=None,
    scaler_y=None,
    conjunto="val"
):
    """
    Predice que el precio futuro del Brent será
    igual al último precio conocido.

    Devuelve las métricas en dólares por barril.

    Si resultados contiene precio_origen_* / precio_real_*
    (preparar_datasets actual) se usan directamente, lo que funciona
    con cualquier objetivo. Si no, se reconstruye desde X (compatibilidad
    con el formato antiguo: el Brent debe ser la columna 0 de X).
    """

    filas = []

    # El Brent ocupa la primera columna de X (solo formato antiguo)
    indice_brent = 0

    for horizonte, datos in resultados.items():

        if f"precio_origen_{conjunto}" in datos:

            y_pred = datos[f"precio_origen_{conjunto}"]
            y_real = datos[f"precio_real_{conjunto}"]

        else:

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
