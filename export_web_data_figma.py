"""Exportación web reproducible, sin red ni dependencias externas.

Lee el CSV del proyecto, resultados publicados y tablas ya ejecutadas del
cuaderno. No entrena redes ni sustituye resultados ausentes por estimaciones.
"""
import csv
import json
import math
import random
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Table(HTMLParser):
    """Extrae tablas simples de pandas; las tablas multiíndice no se infieren."""
    def __init__(self):
        super().__init__()
        self.headers, self.rows, self.row = [], [], []
        self.cell, self.text, self.body = None, "", False

    def handle_starttag(self, tag, attrs):
        if tag == "tbody":
            self.body = True
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell, self.text = tag, ""

    def handle_data(self, data):
        if self.cell:
            self.text += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell:
            if not self.body or self.cell == "td":
                self.row.append(self.text.strip())
            self.cell = None
        if tag == "tr":
            if self.body and self.row:
                self.rows.append(self.row)
            elif self.row:
                self.headers = [x for x in self.row if x]

    def records(self):
        return [dict(zip(self.headers, row)) for row in self.rows
                if len(row) == len(self.headers)]


def quantile(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def business_date(origin, days):
    d = date.fromisoformat(origin)
    for _ in range(days):
        d += timedelta(days=1)
        while d.weekday() > 4:
            d += timedelta(days=1)
    return d.isoformat()


def numeric(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return value


def notebook_records(notebook, marker):
    result = []
    for cell in notebook["cells"]:
        if marker not in "".join(cell.get("source", [])):
            continue
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if html:
                table = Table()
                table.feed("".join(html))
                result.extend({k: numeric(v) for k, v in r.items()}
                              for r in table.records())
    return result


def uncertainty(history):
    """Mismas ecuaciones que src/incertidumbre.py: λ=0,94 y φ=0,99."""
    p = [x["precio"] for x in history]
    r = [math.log(p[i] / p[i - 1]) for i in range(1, len(p))]
    ewma, cumulative, vs, longs = r[0] ** 2, 0, [], []
    for i, ret in enumerate(r):
        ewma = .94 * ewma + .06 * ret ** 2
        cumulative += ret ** 2
        vs.append(ewma)
        longs.append(cumulative / (i + 1))
    vols = [math.sqrt(x * 252) * 100 for x in vs]
    percentile = sum(x < vols[-1] for x in vols) / len(vols) * 100
    out = {}
    for h in (5, 63, 252):
        def sigma(i, horizon):
            weight = (1 - .99 ** horizon) / (horizon * (1 - .99))
            return math.sqrt(horizon * (longs[i - 1] + weight * (vs[i - 1] - longs[i - 1])))
        z = [math.log(p[i + h] / p[i]) / sigma(i, h)
             for i in range(20, len(p) - h) if sigma(i, h) > 0]
        qs = [quantile(z, q) for q in (.1, .25, .75, .9)]
        rng = random.Random(42 + h)
        simulations = [p[-1]] * 3
        points = []
        for step in range(h + 1):
            sig = sigma(len(p) - 1, step) if step else 0
            if step:
                daily = sigma(len(p) - 1, 1)
                simulations = [v * math.exp(rng.gauss(0, daily)) for v in simulations]
            points.append({
                "fecha": business_date(history[-1]["fecha"], step),
                "central": p[-1],
                "banda80": [p[-1] * math.exp(qs[0] * sig), p[-1] * math.exp(qs[3] * sig)],
                "banda50": [p[-1] * math.exp(qs[1] * sig), p[-1] * math.exp(qs[2] * sig)],
                "sim1": simulations[0], "sim2": simulations[1], "sim3": simulations[2],
            })
        out[str(h)] = points
    return {"anual": vols[-1], "percentil": percentile,
            "nivel": "Alta" if percentile >= 67 else "Baja" if percentile <= 33 else "Media"}, out



def blind_test():
    """Carga predicciones cerradas, nunca reentrena ni rellena valores ausentes."""
    folder = ROOT / "resultados/prueba_ciega_2026-11-19"
    result = {"fechaCorte": "2026-11-18", "disponible": False, "predicciones": [], "observaciones": []}
    csvfile = folder / "redes_h1_h2/predicciones.csv"
    if csvfile.exists():
        with csvfile.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    if row.get("Fecha origen") != "2026-11-18":
                        continue
                    lower = next((numeric(v) for k,v in row.items() if k.startswith("Límite inferior")), None)
                    upper = next((numeric(v) for k,v in row.items() if k.startswith("Límite superior")), None)
                    result["predicciones"].append({
                        "fecha": row["Fecha objetivo (aprox.)"],
                        "horizonte": int(row["Horizonte"]),
                        "modelo": row["Modelo"],
                        "precio": float(row["Brent predicho"]),
                        "inferior": lower if isinstance(lower,(int,float)) else None,
                        "superior": upper if isinstance(upper,(int,float)) else None,
                    })
                except (KeyError, ValueError):
                    continue
        result["disponible"] = bool(result["predicciones"])
    # Cotizaciones posteriores al corte: archivo opcional con fecha,precio,tipo,fuente.
    # No se leen automáticamente del dataset, porque podrían ser datos propagados.
    observations = folder / "observaciones.csv"
    if observations.exists():
        with observations.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    if r["fecha"] > result["fechaCorte"] and r["tipo"] and r["fuente"]:
                        result["observaciones"].append({"fecha": r["fecha"], "precio": float(r["precio"]), "tipo":r["tipo"], "fuente":r["fuente"]})
                except (ValueError, KeyError):
                    continue
    return result

def export():
    with (ROOT / "data/petroleo_dataset.csv").open() as f:
        history = [{"fecha": r["Date"], "precio": float(r["brent"])}
                   for r in csv.DictReader(f) if r.get("brent") and float(r["brent"]) > 0]
    notebook = json.loads((ROOT / "notebook/analisis_petroleo.ipynb").read_text())
    metrics = notebook_records(notebook, "# MÉTRICAS EN TEST - 5, 63 Y 252 DÍAS")
    r2 = notebook_records(notebook, "# R² EN TEST: 5, 63 Y 252 DÍAS")
    for row in metrics:
        for other in r2:
            if (row.get("Horizonte") == other.get("Horizonte")
                    and row.get("Modelo") == other.get("Modelo")
                    and (row.get("Modelo") == "Ingenuo" or row.get("Objetivo") == other.get("Objetivo"))):
                row["R²"] = other.get("R²")
    wf = notebook_records(notebook, "# RESUMEN DE RESULTADOS")
    wf = [r for r in wf if "Fold" in r and "MAE" in r]
    calibration = notebook_records(notebook, "# CALIBRACIÓN DEL RANGO DEL 80 %")
    vol, scenarios = uncertainty(history)
    forecasts, retraining = [], None
    runs = list((ROOT / "resultados").glob("reentrenamiento/*/predicciones.csv"))
    if runs:
        latest = max(runs, key=lambda p: date(*reversed([int(x) for x in p.parent.name.split("_")])) )
        config_file = latest.parent / "configuracion.json"
        config = json.loads(config_file.read_text()) if config_file.exists() else {}
        retraining = config.get("fecha_reentrenamiento")
        with latest.open() as f:
            forecasts = [{**{k: numeric(v) for k, v in r.items()},
                          "Objetivo": config.get("objetivo", "retorno")} for r in csv.DictReader(f)]
    curves = {str(h): json.loads((ROOT / f"resultados/curvas_aprendizaje_h{h}.json").read_text())
              for h in (5, 63, 252)}
    data = {"historico": history, "volatilidad": vol, "escenarios": scenarios,
            "pruebaCiega": blind_test(),
            "metricas": metrics, "validacion": wf, "calibracion": calibration,
            "curvas": curves, "predicciones": forecasts, "reentrenamiento": retraining,
            "fuente": "CSV del proyecto · FRED / EIA",
            "fuenteMetricas": "Salidas ejecutadas de analisis_petroleo.ipynb; cifras redondeadas del cuaderno.",
            "horaActualizacion": None}
    target = ROOT / "app/public/datos/brent.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    print(f"Exportadas {len(history)} observaciones, {len(metrics)} métricas y {len(forecasts)} predicciones publicadas.")


if __name__ == "__main__":
    export()
