
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CONFIGURACION
# ==========================================

VARIABLES_ORIGINALES = [
    "brent",
    "wti",
    "gas_natural",
    "tipo_bce",
    "usd_por_eur",
    "cny_por_usd",
    "vix",
    "indice_dolar",
]

VARIABLES_MODELO = VARIABLES_ORIGINALES + [
    "brent_ret",
    "wti_ret",
    "brent_vol_20",
]

HORIZONTES = [5, 63, 252]


# ==========================================
# 2. PREPARACION DE VARIABLES
# ==========================================

def preparar_variables(datos):
    """
    Selecciona las variables y calcula características
    derivadas utilizando únicamente información pasada.
    """

    df = datos[VARIABLES_ORIGINALES].copy()
    df = df.sort_index()

    # Variaciones porcentuales diarias
    df["brent_ret"] = df["brent"].pct_change() * 100
    df["wti_ret"] = df["wti"].pct_change() * 100

    # Volatilidad histórica de 20 observaciones
    df["brent_vol_20"] = (
        df["brent_ret"]
        .rolling(window=20, min_periods=20)
        .std()
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    # Eliminar filas iniciales sin datos suficientes
    df = df.dropna()

    return df


# ==========================================
# 3. DIVISION CRONOLOGICA
# ==========================================

def obtener_limites(df, train_ratio=0.70, val_ratio=0.15):
    """
    Calcula los límites temporales de los conjuntos.
    """

    n = len(df)

    fin_train = int(n * train_ratio)
    fin_val = int(n * (train_ratio + val_ratio))

    return fin_train, fin_val


# ==========================================
# 4. NORMALIZACION
# ==========================================

def normalizar_datos(df, fin_train):
    """
    Ajusta los escaladores exclusivamente
    con observaciones de entrenamiento.
    """

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Ajustar solo con entrenamiento
    scaler_X.fit(df.iloc[:fin_train][VARIABLES_MODELO])
    scaler_y.fit(df.iloc[:fin_train][["brent"]])

    # Transformar el histórico completo
    X_scaled = scaler_X.transform(df[VARIABLES_MODELO])

    y_scaled = scaler_y.transform(df[["brent"]]).ravel()

    return X_scaled, y_scaled, scaler_X, scaler_y


# ==========================================
# 5. CREACION DE SECUENCIAS
# ==========================================

def crear_secuencias(
    X_scaled,
    y_scaled,
    ventana,
    horizonte
):
    """
    Crea ventanas de entrada y objetivos futuros.

    Cada ventana contiene 'ventana' observaciones.
    El objetivo es el Brent 'horizonte' posiciones
    después de la última observación de entrada.
    """

    X, y, indices_objetivo = [], [], []

    n = len(X_scaled)

    for fin in range(ventana, n - horizonte + 1):

        inicio = fin - ventana
        objetivo = fin - 1 + horizonte

        X.append(X_scaled[inicio:fin])
        y.append(y_scaled[objetivo])
        indices_objetivo.append(objetivo)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(indices_objetivo)
    )


# ==========================================
# 6. PREPARACION COMPLETA
# ==========================================

def preparar_datasets(
    datos,
    ventana=60,
    horizontes=HORIZONTES
):
    """
    Prepara los datos para todos los horizontes.

    Mantiene las mismas fechas de división para
    todos los modelos y evita que los objetivos
    crucen los límites de cada conjunto.
    """

    df = preparar_variables(datos)

    fin_train, fin_val = obtener_limites(df)

    X_scaled, y_scaled, scaler_X, scaler_y = (
        normalizar_datos(df, fin_train)
    )

    resultados = {}

    for horizonte in horizontes:

        X, y, indices = crear_secuencias(
            X_scaled,
            y_scaled,
            ventana,
            horizonte
        )

        # Separar según el índice de la etiqueta futura
        mask_train = indices < fin_train

        mask_val = (
            (indices >= fin_train)
            & (indices < fin_val)
        )

        mask_test = indices >= fin_val

        resultados[horizonte] = {
            "X_train": X[mask_train],
            "y_train": y[mask_train],

            "X_val": X[mask_val],
            "y_val": y[mask_val],

            "X_test": X[mask_test],
            "y_test": y[mask_test],

            "fechas_train": df.index[indices[mask_train]],
            "fechas_val": df.index[indices[mask_val]],
            "fechas_test": df.index[indices[mask_test]],
        }

    return {
        "datos": df,
        "resultados": resultados,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "fin_train": fin_train,
        "fin_val": fin_val,
        "variables": VARIABLES_MODELO,
    }
