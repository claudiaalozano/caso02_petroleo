
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
    "oro",
    "cobre"
]

# Enfoque original: precios en nivel + retornos simples
VARIABLES_MODELO = VARIABLES_ORIGINALES + [
    "brent_ret",
    "wti_ret",
    "brent_vol_20",
]

# Enfoque recomendado: variables estacionarias
SERIES_PRECIO = [
    "brent",
    "wti",
    "gas_natural",
    "usd_por_eur",
    "cny_por_usd",
    "indice_dolar",
    "oro",
    "cobre",
]

VARIABLES_RETORNO = [f"r_{c}" for c in SERIES_PRECIO] + [
    "d_tipo_bce",
    "log_vix",
    "brent_vs_ma20",
    "brent_vs_ma60",
    "brent_vs_ma250",
    "brent_vol_20",
    "spread_brent_wti",
]

# Variables de futuros (primer contrato continuo de Yahoo: BZ=F y CL=F).
# El dataset no incluye el segundo contrato, así que la pendiente real de la
# curva (M1 - M2) no está disponible; se usan aproximaciones:
# - base:      log(futuro / spot)  (contango / backwardation a muy corto plazo)
# - rango:     log(máximo / mínimo) intradía del futuro (volatilidad)
# - vol_rel:   log-volumen frente a su media de 20 días (actividad anormal)
VARIABLES_FUTUROS_WTI = ["base_wti", "rango_wti_fut", "vol_rel_wti_fut"]
VARIABLES_FUTUROS_BRENT = ["base_brent", "rango_brent_fut", "vol_rel_brent_fut"]

# futuros=None -> sin variables de futuros
# futuros="wti" -> solo WTI (histórico desde 2000)
# futuros="todos" -> WTI + Brent (BZ=F solo existe desde 2007-07:
#                    el entrenamiento empieza más tarde)
OPCIONES_FUTUROS = (None, "wti", "todos")

OBJETIVOS = ("nivel", "retorno")

HORIZONTES = [5, 63, 252]

# Límite para winsorizar retornos logarítmicos diarios.
# Evita que episodios como el WTI negativo (abril 2020)
# dominen la normalización.
LIMITE_RETORNO = 0.20


# ==========================================
# 2. PREPARACION DE VARIABLES
# ==========================================

def preparar_variables(datos):
    """
    Enfoque original (objetivo = nivel).

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


def variables_futuros(datos, prefijo, spot):
    """Base, rango intradía y volumen relativo del futuro 'prefijo'."""

    cierre = datos[f"{prefijo}_Close"].clip(lower=1)
    log_volumen = np.log1p(datos[f"{prefijo}_Volume"].clip(lower=0))
    nombre = prefijo.replace("_fut", "")

    return pd.DataFrame({
        f"base_{nombre}": np.log(cierre / datos[spot].clip(lower=1)).clip(-0.2, 0.2),
        f"rango_{nombre}_fut": np.log(
            datos[f"{prefijo}_High"].clip(lower=1) / datos[f"{prefijo}_Low"].clip(lower=1)
        ).clip(0, 0.3),
        f"vol_rel_{nombre}_fut": (
            log_volumen - log_volumen.rolling(20).mean()
        ).clip(-3, 3),
    }, index=datos.index)


def lista_variables_retorno(futuros=None):
    """Variables de entrada del enfoque retorno según la opción de futuros."""

    if futuros not in OPCIONES_FUTUROS:
        raise ValueError(f"futuros debe ser uno de {OPCIONES_FUTUROS}")

    variables = list(VARIABLES_RETORNO)

    if futuros in ("wti", "todos"):
        variables += VARIABLES_FUTUROS_WTI

    if futuros == "todos":
        variables += VARIABLES_FUTUROS_BRENT

    return variables


def preparar_variables_retorno(datos, futuros=None):
    """
    Enfoque recomendado (objetivo = retorno).

    Todas las entradas son (aproximadamente) estacionarias:
    - retornos logarítmicos diarios winsorizados,
    - variación diaria del tipo del BCE,
    - logaritmo del VIX,
    - distancia del Brent a sus medias móviles (tendencia),
    - volatilidad de 20 días y diferencial Brent-WTI.

    La columna 'brent' (precio) se conserva solo para construir
    el objetivo y reconstruir precios; NO es variable de entrada.
    """

    d = datos[VARIABLES_ORIGINALES].copy().sort_index()

    def log_precio(columna):
        # clip: el WTI llegó a cotizar negativo en abril de 2020
        return np.log(d[columna].clip(lower=1e-3))

    df = pd.DataFrame(index=d.index)

    for c in SERIES_PRECIO:
        df[f"r_{c}"] = (
            log_precio(c).diff()
            .clip(-LIMITE_RETORNO, LIMITE_RETORNO)
        )

    df["d_tipo_bce"] = d["tipo_bce"].diff()
    df["log_vix"] = np.log(d["vix"])

    log_brent = log_precio("brent")

    for ventana in (20, 60, 250):
        df[f"brent_vs_ma{ventana}"] = (
            log_brent - log_brent.rolling(ventana).mean()
        )

    df["brent_vol_20"] = df["r_brent"].rolling(20).std()

    df["spread_brent_wti"] = (
        (log_brent - log_precio("wti")).clip(-0.5, 0.5)
    )

    variables = lista_variables_retorno(futuros)

    if futuros in ("wti", "todos"):
        df = df.join(variables_futuros(datos.sort_index(), "wti_fut", "wti"))

    if futuros == "todos":
        df = df.join(variables_futuros(datos.sort_index(), "brent_fut", "brent"))

    df["brent"] = d["brent"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=variables + ["brent"])

    columnas = variables + ["brent"]

    # Precio del futuro del Brent (primer contrato): NO es variable de entrada,
    # se usa solo como predicción alternativa para las combinaciones.
    # Vale NaN antes de julio de 2007.
    if "brent_fut_Close" in datos.columns:
        df["brent_fut"] = datos["brent_fut_Close"].reindex(df.index)
        columnas.append("brent_fut")

    return df[columnas]


def preparar_variables_segun_objetivo(datos, objetivo="nivel", futuros=None):
    """Devuelve (df, variables_de_entrada) para el objetivo indicado."""

    if objetivo not in OBJETIVOS:
        raise ValueError(f"objetivo debe ser uno de {OBJETIVOS}")

    if objetivo == "nivel":
        if futuros is not None:
            raise ValueError("Las variables de futuros solo están disponibles con objetivo 'retorno'")
        return preparar_variables(datos), VARIABLES_MODELO

    return preparar_variables_retorno(datos, futuros), lista_variables_retorno(futuros)


def serie_objetivo(df, horizonte, objetivo="nivel"):
    """
    Valor objetivo asociado a cada fila t (la fila del objetivo).

    - nivel:   Brent_t
    - retorno: log(Brent_t / Brent_{t-h})

    Con crear_secuencias, el objetivo de una ventana que termina en
    el origen o es la fila o + h, por lo que el retorno es exactamente
    log(Brent_{o+h} / Brent_o): no usa información futura en la entrada.
    """

    if objetivo == "nivel":
        return df["brent"].to_numpy(dtype=float)

    return np.log(df["brent"]).diff(horizonte).to_numpy(dtype=float)


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


def obtener_fechas_limite(datos, train_ratio=0.70, val_ratio=0.15):
    """
    Fechas de inicio de validación y de test.

    Se calculan siempre sobre el dataset del enfoque original para
    que ambos objetivos (nivel y retorno) usen EXACTAMENTE los mismos
    periodos, aunque el enfoque retorno pierda filas iniciales por
    la media móvil de 250 días.
    """

    ref = preparar_variables(datos)
    fin_train, fin_val = obtener_limites(ref, train_ratio, val_ratio)

    return ref.index[fin_train], ref.index[fin_val]


# ==========================================
# 4. NORMALIZACION
# ==========================================

def normalizar_datos(
    df,
    fin_train,
    variables=VARIABLES_MODELO,
    y=None
):
    """
    Ajusta los escaladores exclusivamente
    con observaciones de entrenamiento.

    y: serie objetivo por fila (ver serie_objetivo). Si es None
    se usa el precio del Brent (comportamiento original).
    """

    if y is None:
        y = df["brent"].to_numpy(dtype=float)

    y = np.asarray(y, dtype=float).reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Ajustar solo con entrenamiento (StandardScaler ignora los NaN)
    scaler_X.fit(df.iloc[:fin_train][variables])
    scaler_y.fit(y[:fin_train])

    # Transformar el histórico completo
    X_scaled = scaler_X.transform(df[variables])

    y_scaled = scaler_y.transform(y).ravel()

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
# 6. RECONSTRUCCION DEL PRECIO
# ==========================================

def reconstruir_precio(pred_norm, scaler_y, precio_origen, objetivo="nivel"):
    """
    Convierte la predicción normalizada en dólares por barril.

    - nivel:   desnormalizar.
    - retorno: precio_origen * exp(retorno_predicho).
    """

    valor = scaler_y.inverse_transform(
        np.asarray(pred_norm, dtype=float).reshape(-1, 1)
    ).ravel()

    if objetivo == "nivel":
        return valor

    return np.asarray(precio_origen, dtype=float) * np.exp(valor)


# ==========================================
# 7. PREPARACION COMPLETA
# ==========================================

def preparar_datasets(
    datos,
    ventana=60,
    horizontes=HORIZONTES,
    objetivo="nivel",
    futuros=None
):
    """
    Prepara los datos para todos los horizontes.

    Mantiene las mismas fechas de división para
    todos los modelos y evita que los objetivos
    crucen los límites de cada conjunto.

    objetivo: "nivel" (original) o "retorno" (recomendado).

    Además de X/y normalizados, cada conjunto incluye en dólares:
    - precio_origen_*: último Brent conocido (= predicción ingenua)
    - precio_real_*:   Brent observado en la fecha objetivo
    """

    df, variables = preparar_variables_segun_objetivo(datos, objetivo, futuros)

    inicio_val, inicio_test = obtener_fechas_limite(datos)
    fin_train = int(df.index.searchsorted(inicio_val))
    fin_val = int(df.index.searchsorted(inicio_test))

    precio = df["brent"].to_numpy(dtype=float)

    scaler_X = None
    resultados = {}
    scalers_y = {}

    for horizonte in horizontes:

        y_serie = serie_objetivo(df, horizonte, objetivo)

        X_scaled, y_scaled, scaler_X, scaler_y = normalizar_datos(
            df, fin_train, variables, y_serie
        )

        X, y, indices = crear_secuencias(
            X_scaled,
            y_scaled,
            ventana,
            horizonte
        )

        origen = indices - horizonte

        # Separar según el índice de la etiqueta futura
        mask_train = indices < fin_train

        mask_val = (
            (indices >= fin_train)
            & (indices < fin_val)
        )

        mask_test = indices >= fin_val

        conjuntos = {}

        for nombre, mask in (
            ("train", mask_train),
            ("val", mask_val),
            ("test", mask_test),
        ):
            conjuntos[f"X_{nombre}"] = X[mask]
            conjuntos[f"y_{nombre}"] = y[mask]
            conjuntos[f"fechas_{nombre}"] = df.index[indices[mask]]
            conjuntos[f"precio_origen_{nombre}"] = precio[origen[mask]]
            conjuntos[f"precio_real_{nombre}"] = precio[indices[mask]]

        conjuntos["scaler_y"] = scaler_y
        resultados[horizonte] = conjuntos
        scalers_y[horizonte] = scaler_y

    return {
        "datos": df,
        "resultados": resultados,
        "scaler_X": scaler_X,
        # Con objetivo "nivel" el escalador de y es el mismo para todos
        # los horizontes; con "retorno" hay uno por horizonte
        # (usar resultados[h]["scaler_y"] o scalers_y[h]).
        "scaler_y": scalers_y[horizontes[0]],
        "scalers_y": scalers_y,
        "fin_train": fin_train,
        "fin_val": fin_val,
        "variables": variables,
        "objetivo": objetivo,
    }
