"""Prueba a ciegas del Brent para 19 y 20 de noviembre de 2026.

Congela el histórico el 18/11/2026, realiza validación temporal interna
para elegir épocas y reentrena en todos los datos anteriores al corte.
Predicciones neuronales experimentales h=1 y h=2: NO están validadas
con los resultados previos de h=5, 63 y 252.
No modifica el reentrenamiento de producción original.
"""

import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.preprocessing import (
    preparar_variables_segun_objetivo,
    serie_objetivo,
    normalizar_datos,
    crear_secuencias,
    reconstruir_precio,
)
from src.modelos import crear_modelo, contar_parametros
from src.entrenamiento import fijar_semilla, entrenar_modelo, predecir_modelo
from src.evaluacion import calcular_metricas, skill_vs_ingenuo
from src.validacion_temporal import preparar_fold, ajustar_epocas_fijas
from src.incertidumbre import volatilidad_horizonte, cuantiles_estandarizados


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "petroleo_dataset.csv"

FECHA_CORTE = pd.Timestamp("2026-11-18")
CARPETA_RESULTADOS = (
    RAIZ / "resultados" / "prueba_ciega_2026-11-19" / "redes_h1_h2"
)

MODELOS = ["RNN", "LSTM", "GRU"]

# Horizontes inéditos para nuestro proyecto: predicciones experimentales.
HORIZONTES = [1, 2]

OBJETIVO = "retorno"
VENTANA = 60

# Fracción final del histórico usada como validación interna
FRACCION_VALIDACION = 0.10

# Configuración congelada (la misma que CONFIG_FINAL del notebook)
CONFIG = {
    "hidden_size_1": 32,
    "hidden_size_2": 16,
    "dropout": 0.2,
    "epocas": 30,
    "batch_size": 32,
    "lr": 0.001,
    "paciencia": 5,
    "delta": 0.5,
    "semillas": [0, 1, 2],
}

# Cuantiles del intervalo de predicción (80 %)
CUANTILES = (0.10, 0.90)

# Comparación experimental: media de redes en ambos horizontes.
# El ingenuo se guarda también, para contrastarlo con el resultado real.
COMBINACION_PRINCIPAL = "Comb. redes+ingenuo+futuro"
HORIZONTES_COMBINACION_FUTURO = []
PRINCIPAL_POR_HORIZONTE = {1: "Comb. redes", 2: "Comb. redes"}


# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================

def guardar_pesos(modelo, ruta):
    """Guarda pesos compatibles con CPU."""
    torch.save(
        {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
        ruta,
    )


def dia_habil_objetivo(fecha, horizonte):
    """Fecha aproximada del objetivo: h días hábiles después."""
    return (pd.Timestamp(fecha) + pd.offsets.BDay(horizonte)).date()


def validacion_interna(df, variables, horizonte, tipo):
    """
    Entrena con early stopping en el último 10 % del histórico.

    Devuelve las mejores épocas por semilla y las predicciones fuera de
    muestra (normalizadas) del ensemble en ese tramo.
    """
    n = len(df)
    inicio = int(n * (1 - FRACCION_VALIDACION))
    interno = preparar_fold(df, variables, inicio, n, horizonte, VENTANA, OBJETIVO)

    epocas, preds = [], []

    for semilla in CONFIG["semillas"]:
        fijar_semilla(semilla)
        modelo = crear_modelo(
            tipo=tipo,
            input_size=len(variables),
            hidden_size_1=CONFIG["hidden_size_1"],
            hidden_size_2=CONFIG["hidden_size_2"],
            dropout=CONFIG["dropout"],
        )
        modelo, historial = entrenar_modelo(
            modelo,
            interno["X_train"], interno["y_train"],
            interno["X_val"], interno["y_val"],
            epocas=CONFIG["epocas"],
            batch_size=CONFIG["batch_size"],
            lr=CONFIG["lr"],
            paciencia=CONFIG["paciencia"],
            semilla=semilla,
            verbose=False,
        )
        epocas.append(int(historial["mejor_epoca"]))
        preds.append(np.asarray(predecir_modelo(modelo, interno["X_val"])))

    # Log-retorno predicho (en unidades reales) del ensemble de semillas
    retorno = interno["scaler_y"].inverse_transform(
        np.mean(preds, axis=0).reshape(-1, 1)
    ).ravel()

    return interno, epocas, retorno


def entrenar_final(df, variables, horizonte, tipo, epocas_por_semilla):
    """
    Reentrena con TODO el histórico y predice desde la última ventana.

    Devuelve los modelos, el log-retorno predicho (media de semillas)
    y los escaladores ajustados con todos los datos.
    """
    n = len(df)
    y_serie = serie_objetivo(df, horizonte, OBJETIVO)

    # Escaladores ajustados con todo el histórico
    X_scaled, y_scaled, scaler_X, scaler_y = normalizar_datos(
        df, n, variables, y_serie
    )

    # Todas las ventanas con objetivo conocido
    X, y, _ = crear_secuencias(X_scaled, y_scaled, VENTANA, horizonte)
    datos_train = {"X_train": X, "y_train": y}

    # Última ventana disponible (su objetivo está en el futuro)
    X_ultima = X_scaled[n - VENTANA:n][np.newaxis].astype(np.float32)

    modelos, preds = [], []

    for semilla, epocas in zip(CONFIG["semillas"], epocas_por_semilla):
        modelo = ajustar_epocas_fijas(
            tipo, datos_train, CONFIG, max(epocas, 1), semilla
        )
        modelos.append(modelo)
        preds.append(float(predecir_modelo(modelo, X_ultima)[0]))

    retorno = float(
        scaler_y.inverse_transform([[np.mean(preds)]])[0, 0]
    )

    return modelos, retorno, scaler_X, scaler_y


# ==========================================
# 3. REENTRENAMIENTO
# ==========================================

def reentrenar_modelos():

    # No sobrescribir una predicción previamente registrada.
    if (CARPETA_RESULTADOS / "predicciones.csv").exists():
        raise FileExistsError(
            f"La prueba ya tiene predicciones: {CARPETA_RESULTADOS}. "
            "No se sobrescriben resultados de una prueba a ciegas."
        )

    datos = pd.read_csv(
        RUTA_DATOS,
        parse_dates=["Date"],
        index_col="Date"
    ).sort_index()
    if datos.index.has_duplicates:
        raise ValueError("Hay fechas duplicadas en el dataset")

    # CRÍTICO: el corte se aplica antes del cálculo de las variables.
    datos = datos.loc[:FECHA_CORTE].copy()
    if datos.empty or datos.index[-1].normalize() != FECHA_CORTE:
        raise ValueError(
            "No hay datos del 18/11/2026 como último registro "
            "del histórico congelado."
        )

    df, variables = preparar_variables_segun_objetivo(datos, OBJETIVO)

    fecha_origen = df.index[-1]
    if fecha_origen.normalize() != FECHA_CORTE:
        raise ValueError(
            f"El preprocesamiento termina el {fecha_origen.date()}, "
            "no el 18/11/2026. Revisar datos y preprocesamiento."
        )

    # Crear la carpeta solo después de validar el corte temporal.
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    precio_actual = float(df["brent"].iloc[-1])
    futuro_actual = float(df["brent_fut"].iloc[-1]) if "brent_fut" in df else np.nan

    print("Última fecha:", fecha_origen.date())
    print("Último Brent:", precio_actual)
    print("Observaciones:", len(df))

    metricas, predicciones, archivos = [], [], []
    epocas_elegidas = {}
    scalers_y = {}
    scaler_X = None

    for horizonte in HORIZONTES:

        print("\n" + "=" * 60)
        print(f"HORIZONTE: {horizonte} DÍAS")
        print("=" * 60)

        retornos_internos = {}
        retornos_futuros = {}
        interno = None

        for tipo in MODELOS:

            # 1. Validación interna (último 10 %)
            interno, epocas, retorno_interno = validacion_interna(
                df, variables, horizonte, tipo
            )
            retornos_internos[tipo] = retorno_interno
            epocas_elegidas[f"{tipo.lower()}_h{horizonte}"] = epocas

            if max(epocas) <= 1:
                print(f"AVISO: {tipo} h={horizonte} elige la época 1 en todas las "
                      "semillas (la validación interna no premia el entrenamiento).")

            # 2. Modelo final con todo el histórico
            modelos, retorno_futuro, scaler_X, scaler_y = entrenar_final(
                df, variables, horizonte, tipo, epocas
            )
            retornos_futuros[tipo] = retorno_futuro
            scalers_y[horizonte] = scaler_y

            for semilla, modelo in zip(CONFIG["semillas"], modelos):
                nombre = f"{tipo.lower()}_h{horizonte}_s{semilla}.pth"
                guardar_pesos(modelo, CARPETA_RESULTADOS / nombre)
                archivos.append(nombre)

            print(f"{tipo}: épocas {epocas} | parámetros {contar_parametros(modelos[0])}")

        origen = interno["precio_origen_val"]
        real = interno["precio_real_val"]

        # Predicciones en precio: validación interna (fuera de muestra) y futuro
        pred_internas = {t: origen * np.exp(retornos_internos[t]) for t in MODELOS}
        pred_futuras = {t: precio_actual * np.exp(retornos_futuros[t]) for t in MODELOS}

        # Combinación con pesos iguales de las tres redes
        pred_internas["Comb. redes"] = origen * np.exp(
            np.mean([retornos_internos[t] for t in MODELOS], axis=0))
        pred_futuras["Comb. redes"] = precio_actual * np.exp(
            np.mean([retornos_futuros[t] for t in MODELOS]))

        # Combinación de futuro DESACTIVADA para h=1 y h=2: 1/3 redes + 1/3 ingenuo
        # + 1/3 futuro del Brent. El futuro es el primer contrato, por lo que no
        # se usa como predicción a 63 o 252 días.
        futuro_interno = interno["futuro_origen_val"]
        if (horizonte in HORIZONTES_COMBINACION_FUTURO
                and np.isfinite(futuro_interno).all() and np.isfinite(futuro_actual)):
            pred_internas[COMBINACION_PRINCIPAL] = (
                pred_internas["Comb. redes"] + origen + futuro_interno) / 3
            pred_futuras[COMBINACION_PRINCIPAL] = (
                pred_futuras["Comb. redes"] + precio_actual + futuro_actual) / 3

        # Modelo ingenuo: referencia de precio constante para h=1 y h=2.
        pred_internas = {"Ingenuo": origen, **pred_internas}
        pred_futuras = {"Ingenuo": precio_actual, **pred_futuras}

        principal = PRINCIPAL_POR_HORIZONTE.get(horizonte, "Comb. redes")
        if principal not in pred_futuras:
            principal = "Comb. redes"

        # Intervalos exploratorios adaptados a volatilidad (src/incertidumbre.py).
        # El método fue estudiado en otros horizontes, NO validado para h1/h2: los cuantiles del error estandarizado se
        # calibran con el histórico ANTERIOR al corte y se aplican con
        # la volatilidad prevista el 18/11/2026. Calibrar solo con el último
        # 10 % daba cuantiles asimétricos y distorsionados por los picos de 2026.
        sigma_h = volatilidad_horizonte(df["brent"], horizonte)
        log_precio = np.log(df["brent"])
        error_hist = (log_precio.shift(-horizonte) - log_precio).dropna()
        q_z = cuantiles_estandarizados(error_hist, sigma_h.reindex(error_hist.index), CUANTILES)
        sigma_actual = float(sigma_h.iloc[-1])

        # 3. Métricas en la validación interna (fuera de muestra)
        mae_ingenuo = calcular_metricas(real, origen)["MAE"]

        for nombre, pred in pred_internas.items():

            m = calcular_metricas(real, pred, origen)
            metricas.append({
                "Horizonte": horizonte, "Modelo": nombre, **m,
                "Skill": skill_vs_ingenuo(m["MAE"], mae_ingenuo),
            })

            # Intervalo adaptado a la volatilidad actual (mismo para todos los modelos)
            q_bajo, q_alto = q_z * sigma_actual

            precio_predicho = float(pred_futuras[nombre])

            predicciones.append({
                "Horizonte": horizonte,
                "Modelo": nombre,
                "Principal": nombre == principal,
                "Fecha origen": fecha_origen.date(),
                "Fecha objetivo (aprox.)": dia_habil_objetivo(fecha_origen, horizonte),
                "Brent actual": precio_actual,
                "Brent predicho": precio_predicho,
                "Variación (%)": 100 * (precio_predicho / precio_actual - 1),
                "Volatilidad prevista (%)": 100 * sigma_actual,
                f"Límite inferior ({int(100 * (CUANTILES[1] - CUANTILES[0]))} %)":
                    precio_predicho * np.exp(q_bajo),
                f"Límite superior ({int(100 * (CUANTILES[1] - CUANTILES[0]))} %)":
                    precio_predicho * np.exp(q_alto),
            })

    # ==========================================
    # 4. GUARDAR RESULTADOS
    # ==========================================

    joblib.dump(scaler_X, CARPETA_RESULTADOS / "scaler_X.pkl")
    # Diccionario {horizonte: StandardScaler}
    joblib.dump(scalers_y, CARPETA_RESULTADOS / "scaler_y.pkl")

    tabla_metricas = pd.DataFrame(metricas)
    tabla_metricas.to_csv(CARPETA_RESULTADOS / "metricas.csv", index=False)

    tabla_predicciones = pd.DataFrame(predicciones)
    tabla_predicciones.to_csv(CARPETA_RESULTADOS / "predicciones.csv", index=False)

    with open(CARPETA_RESULTADOS / "historiales.json", "w", encoding="utf-8") as f:
        json.dump({"mejores_epocas_por_semilla": epocas_elegidas}, f, indent=4)

    archivos += [
        "scaler_X.pkl", "scaler_y.pkl", "metricas.csv",
        "predicciones.csv", "historiales.json", "configuracion.json",
    ]

    configuracion = {
        "fecha_reentrenamiento": "prueba_ciega_2026-11-19",
        "fecha_corte": str(FECHA_CORTE.date()),
        "tipo_prueba": "prospectiva exploratoria h1 h2 (sin validación previa de estas redes)",
        "ultima_fecha_datos": str(fecha_origen.date()),
        "dataset_sha256": hashlib.sha256(datos.to_csv().encode("utf-8")).hexdigest(),
        "entrenamiento": "histórico hasta el 18/11/2026 (épocas elegidas en el último "
                         f"{int(FRACCION_VALIDACION * 100)} %)",
        "objetivo": OBJETIVO,
        "descripcion_objetivo": "log(Brent_{t+h}/Brent_t); precio = Brent_t * exp(pred)",
        "combinacion": {
            "Comb. redes": "media de los log-retornos de RNN, LSTM y GRU",
        },
        "prediccion_principal": {str(h): m for h, m in PRINCIPAL_POR_HORIZONTE.items()},
        "intervalo": (f"cuantiles {CUANTILES} del retorno histórico a h días anterior al corte, "
                      "estandarizado por la volatilidad prevista para el horizonte (EWMA "
                      "λ=0,94 con vuelta a la media φ=0,99) y aplicado con la volatilidad "
                      "actual; ver src/incertidumbre.py"),
        "scaler_y": "dict {horizonte: StandardScaler}",
        "modelos": MODELOS,
        "horizontes": HORIZONTES,
        "ventana": VENTANA,
        "variables": list(variables),
        "numero_variables": len(variables),
        "configuracion": CONFIG,
        "funcion_perdida": "HuberLoss",
        "archivos": archivos,
    }

    with open(CARPETA_RESULTADOS / "configuracion.json", "w", encoding="utf-8") as f:
        json.dump(configuracion, f, indent=4, ensure_ascii=False)

    pd.set_option("display.width", 200)
    print("\nREENTRENAMIENTO FINALIZADO")
    print("\nMétricas en la validación interna (último "
          f"{int(FRACCION_VALIDACION * 100)} %, fuera de muestra):")
    print(tabla_metricas[["Horizonte", "Modelo", "MAE", "Skill", "Sesgo",
                          "Acierto dir. (%)"]].round(3).to_string(index=False))
    print("\nPredicciones:")
    print(tabla_predicciones.round(2).to_string(index=False))
    print("\nResultados guardados en:", CARPETA_RESULTADOS)


if __name__ == "__main__":
    reentrenar_modelos()
