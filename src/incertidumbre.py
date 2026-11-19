"""
Rango de incertidumbre adaptado a la volatilidad del mercado.

El SIGNO de los movimientos del Brent es impredecible (autocorrelación de los
retornos ≈ 0), pero su TAMAÑO no: la volatilidad se agrupa (autocorrelación de
|retornos| ≈ 0,24 a 1 día). Por eso el rango del 80 % se escala con la
volatilidad prevista para el horizonte:

    σ_h = sqrt(h) · sqrt( v_larga + peso_h · (v_ewma − v_larga) )
    peso_h = (1 − φ^h) / (h · (1 − φ))

- v_ewma: varianza diaria reciente (EWMA con λ = 0,94, el estándar RiskMetrics).
- v_larga: varianza diaria media histórica (solo con datos pasados).
- φ = 0,99: velocidad a la que la volatilidad vuelve a su nivel normal
  (vida media de ~70 días). Elegido con datos de 2005-2022, antes del test.

El rango se calibra con errores pasados estandarizados, z = error / σ_h, y se
aplica con la volatilidad actual: [pred · exp(q10(z)·σ_h), pred · exp(q90(z)·σ_h)].
"""
import numpy as np
import pandas as pd

LAMBDA = 0.94
PHI = 0.99
CUANTILES = (0.10, 0.90)


def varianzas(precios):
    """Varianza diaria reciente (EWMA) y de largo plazo, usando solo el pasado."""
    r = np.log(precios).diff()
    v_ewma = (r ** 2).ewm(alpha=1 - LAMBDA, adjust=False).mean()
    v_larga = (r ** 2).expanding(min_periods=20).mean()
    return v_ewma, v_larga


def volatilidad_horizonte(precios, horizonte, phi=PHI):
    """σ_h en cada fecha: desviación prevista del log-retorno a h días."""
    v_ewma, v_larga = varianzas(precios)
    peso = (1 - phi ** horizonte) / (horizonte * (1 - phi))
    return np.sqrt(horizonte * (v_larga + peso * (v_ewma - v_larga)))


def cuantiles_estandarizados(error_log, sigma, cuantiles=CUANTILES):
    """Cuantiles de los errores pasados divididos por su σ_h en el origen."""
    z = np.asarray(error_log, dtype=float) / np.asarray(sigma, dtype=float)
    return np.nanquantile(z, cuantiles)


def intervalo(prediccion, q_z, sigma_actual):
    """Límites del rango para la predicción con la volatilidad actual."""
    return prediccion * np.exp(q_z[0] * sigma_actual), prediccion * np.exp(q_z[1] * sigma_actual)


def estado_volatilidad(precios):
    """Volatilidad actual anualizada y su percentil en la historia."""
    v_ewma, _ = varianzas(precios)
    vol_anual = np.sqrt(v_ewma * 252) * 100
    actual = float(vol_anual.iloc[-1])
    percentil = float((vol_anual.dropna() < actual).mean() * 100)
    nivel = "alta" if percentil >= 67 else ("baja" if percentil <= 33 else "normal")
    return {"vol_anual": actual, "percentil": percentil, "nivel": nivel,
            "vol_mediana": float(vol_anual.median())}


def evaluar_calibracion(precios, horizontes=(5, 63, 252), phis=(None, 0.97, 0.99, 0.995),
                        inicio="2005-01-01", corte_test="2022-12-09", paso=5):
    """
    Cobertura real del rango del 80 % del modelo ingenuo, calibrado en tiempo
    real (solo con errores ya observados), según el régimen de volatilidad.
    phi=None -> rango fijo (sin ajustar a la volatilidad).
    """
    lr = np.log(precios)
    v_ewma, _ = varianzas(precios)
    vol = np.sqrt(v_ewma)
    umbral1 = vol.expanding(252).quantile(1 / 3)
    umbral2 = vol.expanding(252).quantile(2 / 3)
    filas = []
    for h in horizontes:
        e = (lr.shift(-h) - lr).dropna()
        for phi in phis:
            s = (volatilidad_horizonte(precios, h, phi).reindex(e.index) if phi
                 else pd.Series(1.0, index=e.index))
            z = (e / s).to_numpy()
            registros = []
            for i in range(len(e)):
                if e.index[i] < pd.Timestamp(inicio) or i % paso:
                    continue
                pasado = z[: max(i - h + 1, 0)]
                if np.isfinite(pasado).sum() < 500 or not np.isfinite(z[i]):
                    continue
                lo, hi = np.nanquantile(pasado, CUANTILES) * s.iloc[i]
                f = e.index[i]
                reg = ("baja" if vol[f] <= umbral1[f] else "media" if vol[f] <= umbral2[f] else "alta")
                registros.append((f, lo <= e.iloc[i] <= hi, 100 * (np.exp(hi) - np.exp(lo)), reg))
            d = pd.DataFrame(registros, columns=["Fecha", "dentro", "ancho", "regimen"]).set_index("Fecha")
            for periodo, sub in [("Selección 2005-2022", d.loc[:corte_test].iloc[:-1]),
                                 ("Test dic-2022 - 2026", d.loc[corte_test:])]:
                fila = {"Horizonte": h, "Método": "Fijo" if phi is None else f"Adaptativo (φ={phi})",
                        "Periodo": periodo, "Cobertura (%)": 100 * sub.dentro.mean(),
                        "Ancho medio (%)": sub.ancho.mean()}
                for reg in ("baja", "media", "alta"):
                    fila[f"Cobertura vol. {reg} (%)"] = 100 * sub[sub.regimen == reg].dentro.mean()
                fila["Error de calibración (pp)"] = np.mean(
                    [abs(fila[f"Cobertura vol. {r} (%)"] - 80) for r in ("baja", "media", "alta")])
                filas.append(fila)
    return pd.DataFrame(filas)
