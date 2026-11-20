"""Exporta la serie histórica real para la web. Ejecutar: python scripts/exportar_historial.py"""
import json
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
origen = RAIZ / 'data' / 'petroleo_dataset.csv'
salida = RAIZ / 'app' / 'historico.json'
df = pd.read_csv(origen, parse_dates=['Date']).sort_values('Date')
df['brent'] = pd.to_numeric(df['brent'], errors='coerce')
df = df.dropna(subset=['Date', 'brent'])
if df.empty:
    raise ValueError('No hay fechas/precios Brent válidos en el dataset.')
registros = [{'fecha': fecha.strftime('%Y-%m-%d'), 'precio': round(float(precio), 5)} for fecha, precio in zip(df['Date'], df['brent'])]
salida.parent.mkdir(parents=True, exist_ok=True)
salida.write_text(json.dumps({'historico': registros, 'nota': 'Datos históricos del proyecto, no cotización en directo'}, ensure_ascii=False), encoding='utf-8')
print(f'Exportadas {len(registros)} observaciones a {salida}')
