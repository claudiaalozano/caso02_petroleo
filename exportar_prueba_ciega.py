"""Exporta la prueba a ciegas, sin modificar predicciones ni entrenar.
Ejecutar desde la raíz del proyecto: python exportar_prueba_ciega.py
"""
import json
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent
BASE = RAIZ / "resultados" / "prueba_ciega_2026-11-19"
HIST = BASE / "datos_congelados.csv"
PRED = BASE / "redes_h1_h2" / "predicciones.csv"
SALIDA = RAIZ / "docs" / "prueba_ciega.json"

for f in (HIST, PRED):
    if not f.is_file():
        raise FileNotFoundError(f"Falta {f}; no se exportan datos inventados.")

hist = pd.read_csv(HIST, parse_dates=["Date"]).sort_values("Date")
if "brent" not in hist:
    raise KeyError("Falta la columna brent en los datos congelados")
hist = hist.dropna(subset=["brent"]).tail(10)
pred = pd.read_csv(PRED, parse_dates=["Fecha origen", "Fecha objetivo (aprox.)"])
pred = pred.loc[pred["Modelo"].eq("Comb. redes")].sort_values("Horizonte")
if set(pred["Horizonte"].astype(int)) != {1, 2}:
    raise ValueError("Se requieren las dos predicciones h=1 y h=2 de Comb. redes")
inf = next(c for c in pred if c.startswith("Límite inferior"))
sup = next(c for c in pred if c.startswith("Límite superior"))
fecha_corte = pred["Fecha origen"].dt.strftime("%Y-%m-%d").unique()
if len(fecha_corte) != 1 or fecha_corte[0] != "2026-11-18":
    raise ValueError("La prueba no tiene el corte esperado del 18/11/2026")
if hist["Date"].max().strftime("%Y-%m-%d") != fecha_corte[0]:
    raise ValueError("El histórico congelado no termina en la fecha de corte")

salida = {
    "fecha_corte": fecha_corte[0],
    "nota": "Las predicciones se generaron con datos hasta la fecha de corte. Los intervalos son nominales del 80 %.",
    "historico": [
        {"fecha": r.Date.strftime("%Y-%m-%d"), "precio": round(float(r.brent), 5)}
        for r in hist.itertuples(index=False)
    ],
    "predicciones": [
        {
            "fecha": r["Fecha objetivo (aprox.)"].strftime("%Y-%m-%d"),
            "horizonte": int(r["Horizonte"]),
            "modelo": "Comb. redes",
            "precio": round(float(r["Brent predicho"]), 5),
            "inferior": round(float(r[inf]), 5),
            "superior": round(float(r[sup]), 5),
        }
        for _, r in pred.iterrows()
    ],
    "observados": [],
}

# Opcional: precios posteriores al corte, mantenidos aparte del histórico congelado.
# CSV columnas: fecha,precio,tipo,fuente
# tipo: intradia / cierre (el tipo intradia NO se usa como cierre definitivo)
OBS = BASE / "observaciones_web.csv"
if OBS.exists():
    obs = pd.read_csv(OBS)
    if not {"fecha", "precio", "tipo", "fuente"}.issubset(obs.columns):
        raise ValueError("observaciones_web.csv requiere fecha,precio,tipo,fuente")
    fechas_obj = {p["fecha"] for p in salida["predicciones"]}
    for _, r in obs.iterrows():
        fecha = pd.Timestamp(r["fecha"]).strftime("%Y-%m-%d")
        tipo = str(r["tipo"]).strip().lower()
        if fecha not in fechas_obj or tipo not in ("intradia", "cierre"):
            raise ValueError(f"Observación inválida: {fecha}, {tipo}")
        salida["observados"].append({
            "fecha": fecha,
            "precio": float(r["precio"]),
            "tipo": tipo,
            "fuente": str(r["fuente"]),
        })

SALIDA.parent.mkdir(parents=True, exist_ok=True)
SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
print("Prueba a ciegas exportada:", SALIDA)
print("Predicciones:", len(salida["predicciones"]), "Observaciones:", len(salida["observados"]))
