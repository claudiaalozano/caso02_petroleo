"""Ejecutar desde una celda del notebook, después de construir tabla_test.

    exec(open(str(RAIZ / 'exportar_resultados.py'), encoding='utf-8').read())

O copiar este script en una celda. No reentrena: solo exporta métricas.
"""
import json
from pathlib import Path
import pandas as pd

if 'tabla_test' not in globals():
    raise RuntimeError('Ejecuta primero la celda que crea tabla_test.')

raiz_export = Path(RAIZ) if 'RAIZ' in globals() else Path.cwd()
salida_export = raiz_export / 'docs' / 'resultados.json'
salida_export.parent.mkdir(parents=True, exist_ok=True)

resultado_export = {}
for horizonte_export in (5, 63, 252):
    df_export = tabla_test[tabla_test['Horizonte'] == horizonte_export].copy()
    df_export = df_export.drop_duplicates(subset=['Objetivo', 'Modelo'])
    filas_export = []
    for _, fila_export in df_export.iterrows():
        def numero(col):
            valor = fila_export.get(col)
            return None if pd.isna(valor) else float(valor)
        filas_export.append({
            'Objetivo': '—' if fila_export['Modelo'] == 'Ingenuo' else str(fila_export['Objetivo']),
            'Modelo': str(fila_export['Modelo']),
            'MAE': numero('MAE'),
            'Skill': None if fila_export['Modelo'] == 'Ingenuo' else numero('Skill'),
            'Sesgo': numero('Sesgo'),
            'Direccion': numero('Acierto dir. (%)'),
            'DM': numero('DM p-valor'),
        })
    resultado_export[str(horizonte_export)] = filas_export
salida_export.write_text(json.dumps(resultado_export, ensure_ascii=False, indent=2), encoding='utf-8')
print('Resultados web actualizados:', salida_export)
