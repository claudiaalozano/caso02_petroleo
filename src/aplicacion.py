"""Lógica de la aplicación: predicciones, escenarios, fiabilidad, textos y ciclo económico."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.incertidumbre import PHI, varianzas, estado_volatilidad

RAIZ = Path(__file__).resolve().parent.parent

HORIZONTES = {
    5: {"nombre": "Corto plazo", "plazo": "una semana", "plazo_corto": "1 semana"},
    63: {"nombre": "Medio plazo", "plazo": "unos 3 meses", "plazo_corto": "3 meses"},
    252: {"nombre": "Largo plazo", "plazo": "un año", "plazo_corto": "1 año"},
}

NOMBRES_MODELO = {
    "Ingenuo": "Precio actual (modelo ingenuo)",
    "Comb. redes+ingenuo+futuro": "Combinación: redes + precio actual + futuro del Brent",
    "Comb. redes": "Combinación de las tres redes",
    "RNN": "RNN", "LSTM": "LSTM", "GRU": "GRU",
}


# ==========================================
# 1. DATOS Y PREDICCIONES
# ==========================================

def cargar_datos():
    return pd.read_csv(RAIZ / "data" / "petroleo_dataset.csv",
                       parse_dates=["Date"], index_col="Date").sort_index()


def ultimo_reentrenamiento():
    """Carpeta más reciente de resultados/reentrenamiento con predicciones."""
    base = RAIZ / "resultados" / "reentrenamiento"
    carpetas = [c for c in base.iterdir() if (c / "predicciones.csv").exists()] if base.exists() else []
    if not carpetas:
        return None
    return max(carpetas, key=lambda c: (c / "predicciones.csv").stat().st_mtime)


def cargar_predicciones(carpeta):
    pred = pd.read_csv(carpeta / "predicciones.csv",
                       parse_dates=["Fecha origen", "Fecha objetivo (aprox.)"])
    pred = pred.rename(columns={
        [c for c in pred.columns if c.startswith("Límite inferior")][0]: "Límite inferior",
        [c for c in pred.columns if c.startswith("Límite superior")][0]: "Límite superior",
    })
    return pred


def cargar_comparativa():
    ruta = RAIZ / "resultados" / "comparativa" / "tabla_comparativa.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


# ==========================================
# 2. ESCENARIOS SIMULADOS (fan chart)
# ==========================================

def simular_escenarios(precios, precio_actual, puntos, dias=252, n=2000, bloque=20,
                       anios=10, semilla=42):
    """
    Trayectorias diarias posibles por simulación histórica filtrada (FHS):

    1. Se estandarizan los retornos históricos por la volatilidad EWMA del día
       anterior (z = r / σ), lo que elimina las épocas agitadas o tranquilas.
    2. Se remuestrean por bloques de 20 días (conservan rachas).
    3. Se reescalan con la volatilidad prevista día a día: parte de la actual
       y vuelve poco a poco a la de largo plazo (φ de src/incertidumbre.py).
    4. Se centran para que la mediana pase por la predicción de cada horizonte.

    puntos: dict {horizonte: precio predicho}
    Devuelve (escenarios [n x dias], cuantiles dict).
    """
    rng = np.random.default_rng(semilla)
    r = np.log(precios).diff()
    v_ewma, v_larga = varianzas(precios)
    z = (r / np.sqrt(v_ewma.shift(1))).dropna().iloc[-252 * anios:].to_numpy()
    z = z / z.std()

    k = np.arange(1, dias + 1)
    v_hoy, v_lp = float(v_ewma.iloc[-1]), float(v_larga.iloc[-1])
    sigma_dia = np.sqrt(v_lp + PHI ** k * (v_hoy - v_lp))

    n_bloques = int(np.ceil(dias / bloque))
    inicios = rng.integers(0, len(z) - bloque, size=(n, n_bloques))
    caminos = np.stack([np.concatenate([z[i:i + bloque] for i in fila])[:dias] for fila in inicios])
    log_caminos = np.cumsum(caminos * sigma_dia, axis=1)
    log_caminos -= np.median(log_caminos, axis=0)

    h = np.r_[0, sorted(puntos)]
    log_pred = np.log(np.r_[precio_actual, [puntos[k_] for k_ in sorted(puntos)]] / precio_actual)
    tendencia = np.interp(k, h, log_pred)
    escenarios = precio_actual * np.exp(log_caminos + tendencia)
    cuantiles = {q: np.percentile(escenarios, q, axis=0) for q in (10, 25, 50, 75, 90)}
    return escenarios, cuantiles


def escenarios_representativos(escenarios, horizonte, percentiles=(20, 40, 60, 80)):
    """Índices de escenarios que terminan cerca de los percentiles dados en el horizonte
    (ejemplos típicos, no extremos)."""
    final = escenarios[:, horizonte - 1]
    objetivos = np.percentile(final, percentiles)
    return [int(np.argmin(np.abs(final - v))) for v in objetivos]


# ==========================================
# 3. FIABILIDAD Y TEXTOS
# ==========================================

def fiabilidad(comparativa, horizonte, modelo_principal):
    """Resumen de la comparativa (validación y test) para el texto y la tabla."""
    if comparativa is None:
        return None
    d = comparativa[comparativa["Horizonte"] == horizonte]
    redes = d[d["Modelo"].isin(["RNN", "LSTM", "GRU"])]
    fila = d[d["Modelo"] == modelo_principal]
    return {
        "mejora_test": float(fila["Mejora test (%)"].iloc[0]) if not fila.empty else 0.0,
        "acierto_test": (float(fila["Acierto dir. test (%)"].iloc[0])
                         if not fila.empty and pd.notna(fila["Acierto dir. test (%)"].iloc[0]) else None),
        "redes_test_min": float(redes["Mejora test (%)"].min()),
        "redes_test_max": float(redes["Mejora test (%)"].max()),
        "redes_val_min": float(redes["Mejora val. (%)"].min()),
        "redes_val_max": float(redes["Mejora val. (%)"].max()),
    }


def nivel_confianza(horizonte):
    return {5: "Moderada", 63: "Baja", 252: "Muy baja"}[horizonte]


def euros(x):
    return f"{x:,.2f} $".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, signo=True):
    return (f"{x:+.1f} %" if signo else f"{x:.1f} %").replace(".", ",")


FACTORES_RIESGO = [
    "**Decisiones de la OPEP+** sobre cuotas de producción, que pueden mover el precio un 5-10 % en días.",
    "**Conflictos geopolíticos** en zonas productoras o rutas de transporte (Oriente Medio, Rusia, estrecho de Ormuz).",
    "**Recesiones o frenazos de la demanda** (China, Estados Unidos), como en 2008 o en la pandemia de 2020.",
    "**Movimientos fuertes del dólar**: el petróleo cotiza en dólares y un dólar fuerte suele abaratarlo.",
    "**Sorpresas en los inventarios** de crudo o en la producción de esquisto de Estados Unidos.",
]


def texto_explicativo(horizonte, fila_principal, fiab, volatilidad=None):
    """Texto breve para un usuario no técnico."""
    p = HORIZONTES[horizonte]["plazo"]
    actual, pred = fila_principal["Brent actual"], fila_principal["Brent predicho"]
    bajo, alto = fila_principal["Límite inferior"], fila_principal["Límite superior"]
    var = fila_principal["Variación (%)"]

    if horizonte == 5:
        que = (f"Para dentro de **{p}**, la previsión es que el Brent esté en torno a "
               f"**{euros(pred)}** ({pct(var)} respecto al precio actual de {euros(actual)}). "
               f"Con un 80 % de probabilidad estará entre **{euros(bajo)} y {euros(alto)}**.")
        if fiab:
            fiable = (f"**Fiabilidad moderada.** En las pruebas con datos de 2022 a 2026, que el modelo no "
                      f"había visto, esta previsión fue un {pct(fiab['mejora_test'], False)} más precisa que "
                      f"suponer que el precio no cambia, y acertó si el precio subía o bajaba el "
                      f"{fiab['acierto_test']:.0f} % de las veces. Es una ayuda para decidir, no una certeza: "
                      f"a una semana vista el petróleo es muy difícil de predecir y el rango es la información "
                      f"más útil.")
        else:
            fiable = "**Fiabilidad moderada.** A una semana vista el petróleo es muy difícil de predecir."
    else:
        que = (f"Para dentro de **{p}**, ningún modelo ha demostrado predecir mejor que el precio actual, "
               f"así que la previsión central es el **precio de hoy, {euros(actual)}**. Lo importante es el "
               f"rango: con un 80 % de probabilidad el Brent estará entre **{euros(bajo)} y {euros(alto)}**.")
        if fiab:
            fiable = (f"**Fiabilidad {nivel_confianza(horizonte).lower()}.** Las redes neuronales parecían "
                      f"acertar más en el periodo 2019-2022 (hasta un {pct(fiab['redes_val_max'], False)} "
                      f"mejor), pero con datos nuevos de 2022 a 2026 fueron entre un "
                      f"{pct(-fiab['redes_test_max'], False)} y un {pct(-fiab['redes_test_min'], False)} peores "
                      f"que no predecir nada. A este plazo el precio depende de acontecimientos que no se "
                      f"pueden anticipar con datos históricos: úsalo para dimensionar el riesgo, no para "
                      f"apostar por una dirección.")
        else:
            fiable = f"**Fiabilidad {nivel_confianza(horizonte).lower()}.**"
    if volatilidad:
        frase = {
            "alta": "El mercado está **más agitado de lo normal**",
            "baja": "El mercado está **más tranquilo de lo normal**",
            "normal": "El mercado está en un nivel de **agitación normal**",
        }[volatilidad["nivel"]]
        efecto = {"alta": "por eso el rango es más ancho que de costumbre",
                  "baja": "por eso el rango es más estrecho que de costumbre",
                  "normal": "así que el rango tiene una anchura habitual"}[volatilidad["nivel"]]
        que += (f" {frase} (volatilidad anual del {volatilidad['vol_anual']:.0f} % frente al "
                f"{volatilidad['vol_mediana']:.0f} % habitual): {efecto}.")
    return que, fiable


# ==========================================
# 4. CICLO ECONÓMICO
# ==========================================

# Cada evento busca el máximo o mínimo del Brent dentro de su ventana de fechas.
# Para los movimientos de 2026, completa el campo "causa" con el acontecimiento
# que los provocó: la aplicación solo describe lo que muestran los datos.
EVENTOS = [
    {"titulo": "Máximo histórico de 2008", "tipo": "max", "desde": "2008-01-01", "hasta": "2008-12-31",
     "causa": "Fuerte crecimiento de la demanda de los países emergentes (sobre todo China), oferta "
              "ajustada y entrada de capital especulativo en materias primas llevaron el Brent a su "
              "máximo histórico en julio de 2008."},
    {"titulo": "Desplome por la crisis financiera", "tipo": "min", "desde": "2008-07-01", "hasta": "2009-06-30",
     "causa": "La quiebra de Lehman Brothers y la recesión mundial hundieron la demanda: el precio cayó "
              "más de un 75 % en apenas cinco meses."},
    {"titulo": "Caída 2014-2016 por exceso de oferta", "tipo": "min", "desde": "2015-01-01", "hasta": "2016-12-31",
     "causa": "El auge del petróleo de esquisto en Estados Unidos y la decisión de la OPEP de no recortar "
              "producción (noviembre de 2014) provocaron un exceso de oferta que llevó el Brent por debajo "
              "de 30 $ en enero de 2016."},
    {"titulo": "Desplome por la pandemia (2020)", "tipo": "min", "desde": "2020-01-01", "hasta": "2020-12-31",
     "causa": "Los confinamientos de la COVID-19 paralizaron el transporte y la demanda se desplomó; la "
              "guerra de precios entre Arabia Saudí y Rusia agravó la caída. El WTI llegó a cotizar en "
              "negativo el 20 de abril de 2020."},
    {"titulo": "Subida tras la invasión de Ucrania (2022)", "tipo": "max", "desde": "2022-01-01", "hasta": "2022-12-31",
     "causa": "La invasión rusa de Ucrania (24 de febrero de 2022) y las sanciones al petróleo ruso, sobre "
              "una demanda que se recuperaba de la pandemia, dispararon el precio por encima de 130 $."},
    {"titulo": "Primer pico de 2026", "tipo": "max", "desde": "2026-01-01", "hasta": "2026-06-30",
     "causa": ""},
    {"titulo": "Segundo pico de 2026", "tipo": "max", "desde": "2026-08-01", "hasta": "2026-12-31",
     "causa": ""},
]


def eventos_ciclo(precios):
    """Fecha y precio de cada evento, calculados sobre los datos."""
    filas = []
    for ev in EVENTOS:
        s = precios.loc[ev["desde"]:ev["hasta"]]
        if s.empty:
            continue
        fecha = s.idxmax() if ev["tipo"] == "max" else s.idxmin()
        causa = ev["causa"] or ("Movimiento reciente de la serie: el Brent alcanzó "
                                f"{euros(s.loc[fecha])} el {fecha:%d/%m/%Y}. "
                                "(Causa pendiente de documentar.)")
        filas.append({**ev, "fecha": fecha, "precio": float(s.loc[fecha]), "causa": causa})
    return filas
