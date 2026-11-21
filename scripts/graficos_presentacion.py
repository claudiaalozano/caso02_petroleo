
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path.cwd()

# Si ejecutas el notebook desde otra carpeta, ajusta RAIZ
PREDICCIONES = RAIZ / "resultados/test_final/predicciones_nivel_h5.csv"
PRECIOS = RAIZ / "resultados/prueba_ciega_2026-11-19/datos_congelados.csv"

SALIDA = RAIZ / "resultados/comparativa"
SALIDA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 17,
    "axes.labelsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

# ============================================================
# 1. CARGAR PRECIOS REALES
# ============================================================

precios = pd.read_csv(PRECIOS)
precios["Date"] = pd.to_datetime(precios["Date"]).dt.normalize()
precios["brent"] = pd.to_numeric(precios["brent"], errors="coerce")

precios = (
    precios[["Date", "brent"]]
    .dropna()
    .drop_duplicates("Date", keep="last")
    .sort_values("Date")
)

# ============================================================
# GRÁFICO 1: PRUEBA A CIEGAS + COBERTURA OBSERVADA
# ============================================================

historico = precios.loc[
    precios["Date"].between("2026-11-05", "2026-11-18")
].copy()

if historico.empty:
    raise ValueError("No se encontró el histórico de noviembre.")

# Valores congelados de la prueba a ciegas.
# Cotizaciones posteriores: futuros Brent, NO spot.
fechas_futuras = pd.to_datetime(["2026-11-19", "2026-11-20"])
predichos = np.array([92.77, 92.43])
inferiores = np.array([87.19, 84.51])
superiores = np.array([98.62, 100.69])
futuros_observados = np.array([91.98, 90.81])

cobertura = (
    (futuros_observados >= inferiores)
    & (futuros_observados <= superiores)
)

porcentaje_cobertura = 100 * cobertura.mean()
errores = np.abs(predichos - futuros_observados)
mae = errores.mean()

fig, ax = plt.subplots(figsize=(14, 7.9))

# Histórico
ax.plot(
    historico["Date"],
    historico["brent"],
    color="#2878c8",
    marker="o",
    linewidth=2.6,
    markersize=5,
    label="Brent histórico (spot)"
)

# Conectar último precio conocido con predicciones
fecha_corte = pd.Timestamp("2026-11-18")
precio_corte = precios.loc[
    precios["Date"] == fecha_corte, "brent"
].iloc[0]

ax.plot(
    [fecha_corte, *fechas_futuras],
    [precio_corte, *predichos],
    color="#e66b39",
    marker="o",
    linestyle="--",
    linewidth=2.7,
    markersize=7,
    label="Predicción combinada"
)

# Intervalos predictivos
ax.fill_between(
    fechas_futuras,
    inferiores,
    superiores,
    color="#f4a38b",
    alpha=0.23,
    label="Intervalo nominal del 80 %"
)

# Observaciones de futuros
ax.scatter(
    fechas_futuras,
    futuros_observados,
    marker="D",
    s=105,
    color="#0c9972",
    edgecolors="white",
    linewidth=1.2,
    zorder=5,
    label="Futuros Brent observados"
)

# Fecha de corte
ax.axvline(
    fecha_corte,
    linestyle=":",
    color="#666666",
    linewidth=1.7,
    label="Fecha de corte: 18/11"
)

# Valores observados
for fecha, precio in zip(fechas_futuras, futuros_observados):
    ax.annotate(
        f"{precio:.2f} USD",
        (fecha, precio),
        xytext=(0, -23),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#087b60"
    )

# Indicadores
texto_resumen = (
    f"COBERTURA OBSERVADA: {porcentaje_cobertura:.0f} %\n"
    f"{int(cobertura.sum())} de {len(cobertura)} valores dentro del intervalo\n"
    f"MAE frente a futuros: {mae:.2f} USD"
)

ax.text(
    0.025, 0.035,
    texto_resumen,
    transform=ax.transAxes,
    fontsize=11,
    va="bottom",
    bbox={
        "boxstyle": "round,pad=0.8",
        "facecolor": "#eaf5f0",
        "edgecolor": "#a1d0bd"
    }
)

ax.set_title(
    "Prueba a ciegas del Brent",
    fontsize=19,
    fontweight="bold",
    pad=22
)

ax.set_ylabel("Precio (USD/barril)")
ax.set_xlabel("Fecha")
ax.grid(alpha=0.18)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02),
          ncol=3, frameon=False)
ax.set_ylim(
    min(inferiores.min(), historico["brent"].min()) - 1.5,
    max(superiores.max(), historico["brent"].max()) + 2.5
)

fig.text(
    0.5, 0.015,
    "19/11: futuro Brent de cierre. 20/11: cotización provisional. "
    "Las predicciones se obtuvieron con Brent spot; "
    "la cobertura de futuros no valida la calibración spot.",
    ha="center",
    fontsize=9,
    color="#666666"
)

fig.tight_layout(rect=[0, 0.045, 1, 0.99])

ruta1 = SALIDA / "prueba_ciegas_cobertura_100.png"
fig.savefig(ruta1, bbox_inches="tight")
plt.show()

print("Gráfico 1 guardado en:", ruta1)

# ============================================================
# GRÁFICO 2: TODOS LOS PUNTOS PREDICHO VS REAL
# ============================================================

pred = pd.read_csv(PREDICCIONES)
pred["Fecha"] = pd.to_datetime(pred["Fecha"]).dt.normalize()

modelos = ["RNN", "LSTM", "GRU"]

# Utilizamos el precio real del dataset congelado,
# no la columna Real del CSV de predicciones.
datos = pred.merge(
    precios,
    left_on="Fecha",
    right_on="Date",
    how="inner",
    validate="many_to_one"
)

for modelo in modelos:
    datos[modelo] = pd.to_numeric(datos[modelo], errors="coerce")

datos = (
    datos.dropna(subset=["brent", *modelos])
    .sort_values("Fecha")
    .drop_duplicates("Fecha", keep="last")
)

if datos.empty:
    raise ValueError("No existen predicciones alineadas con los precios reales.")

colores = {
    "RNN": "#9770b2",
    "LSTM": "#3698b6",
    "GRU": "#d49435"
}

fig, ax = plt.subplots(figsize=(12.7, 8.2))

valores = np.concatenate([
    datos["brent"].to_numpy(),
    *[datos[m].to_numpy() for m in modelos]
])

lim_inf = float(np.floor(valores.min() - 5))
lim_sup = float(np.ceil(valores.max() + 5))

# Recta de predicción perfecta
ax.plot(
    [lim_inf, lim_sup],
    [lim_inf, lim_sup],
    color="#27313c",
    linestyle="--",
    linewidth=2.2,
    label="Predicción perfecta (y = x)",
    zorder=5
)

# Puntos individuales de las tres arquitecturas
for modelo in modelos:
    mae_modelo = np.mean(np.abs(datos[modelo] - datos["brent"]))

    ax.scatter(
        datos["brent"],
        datos[modelo],
        s=19,
        alpha=0.24,
        color=colores[modelo],
        edgecolors="none",
        label=f"{modelo} · MAE {mae_modelo:.2f} USD"
    )

ax.set_xlim(lim_inf, lim_sup)
ax.set_ylim(lim_inf, lim_sup)
ax.set_aspect("equal", adjustable="box")

ax.set_title(
    "Predicción frente a precio real del Brent",
    fontsize=18,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Brent real (USD/barril)")
ax.set_ylabel("Brent predicho (USD/barril)")
ax.grid(alpha=0.17)
ax.legend(loc="upper left", frameon=True, fontsize=10)

ax.text(
    0.98, 0.03,
    f"{len(datos):,} fechas evaluadas\n"
    "Test final · NIVEL · Horizonte 5 días",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox={
        "boxstyle": "round,pad=0.6",
        "facecolor": "#f1f4f7",
        "edgecolor": "#d3dce3"
    }
)

fig.tight_layout()

ruta2 = SALIDA / "scatter_predicho_vs_real_h5.png"
fig.savefig(ruta2, bbox_inches="tight")
plt.show()

print("Gráfico 2 guardado en:", ruta2)

print("\nRESUMEN")
print("Fechas evaluadas:", len(datos))

for modelo in modelos:
    mae_modelo = np.mean(np.abs(datos[modelo] - datos["brent"]))
    print(f"{modelo}: MAE = {mae_modelo:.2f} USD")
