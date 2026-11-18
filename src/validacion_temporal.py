"""Walk-forward con ventana expansiva y test reservado para Brent.

Admite dos formulaciones del objetivo:
- "nivel":   la red predice el precio del Brent (enfoque original).
- "retorno": la red predice log(Brent_{t+h} / Brent_t) y el precio se
             reconstruye como Brent_t * exp(retorno). Recomendado.

Los folds se definen por FECHAS (las del enfoque original), de modo que
ambos objetivos se evalúan exactamente en los mismos periodos.
"""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import (
    preparar_variables_segun_objetivo, obtener_fechas_limite,
    normalizar_datos, crear_secuencias, serie_objetivo, reconstruir_precio,
)
from src.modelos import crear_modelo, contar_parametros
from src.entrenamiento import fijar_semilla, entrenar_modelo, predecir_modelo
from src.evaluacion import (
    calcular_metricas, skill_vs_ingenuo, test_diebold_mariano,
    contraer_prediccion, elegir_alfa,
)


def preparar_bloques(datos, n_folds=5, ventana=60, objetivo="nivel", futuros=None):
    df, variables = preparar_variables_segun_objetivo(datos, objetivo, futuros)
    if not df.index.is_unique:
        raise ValueError('Hay fechas duplicadas')
    fecha_val, fecha_test = obtener_fechas_limite(datos)
    inicio_val = int(df.index.searchsorted(fecha_val))
    inicio_test = int(df.index.searchsorted(fecha_test))
    if n_folds < 2 or inicio_test - inicio_val < n_folds:
        raise ValueError('Número de folds no válido')
    limites = np.linspace(inicio_val, inicio_test, n_folds + 1, dtype=int)
    return df, variables, limites, inicio_test


def preparar_fold(df, variables, inicio, fin, horizonte, ventana=60, objetivo="nivel"):
    # Los objetivos de entrenamiento deben conocerse al emitir la primera predicción.
    corte = inicio - horizonte + 1
    if not (ventana + horizonte < corte <= inicio < fin <= len(df)):
        raise ValueError('No hay suficiente histórico para este bloque')
    sub = df.iloc[:fin]
    y_serie = serie_objetivo(sub, horizonte, objetivo)
    Xs, ys, sx, sy = normalizar_datos(sub, corte, variables, y_serie)
    X, y, idx = crear_secuencias(Xs, ys, ventana, horizonte)
    train = idx < corte
    val = (idx >= inicio) & (idx < fin)
    assert train.any() and val.any()
    assert idx[train].max() <= idx[val].min() - horizonte
    precio = df['brent'].to_numpy(dtype=float)
    origen = idx - horizonte
    futuro = (df['brent_fut'].to_numpy(dtype=float) if 'brent_fut' in df.columns
              else np.full(len(df), np.nan))
    return dict(X_train=X[train], y_train=y[train], X_val=X[val], y_val=y[val],
                fechas_train=df.index[idx[train]], fechas_val=df.index[idx[val]],
                precio_origen_val=precio[origen[val]], precio_real_val=precio[idx[val]],
                futuro_origen_val=futuro[origen[val]],
                fecha_origen=df.index[inicio - horizonte], scaler_X=sx, scaler_y=sy)


def tabla_folds(datos, horizonte=5, n_folds=5, ventana=60, objetivo="nivel", futuros=None):
    df, variables, limites, inicio_test = preparar_bloques(datos, n_folds, ventana, objetivo,
                                                           futuros)
    filas = []
    for k, (a, b) in enumerate(zip(limites[:-1], limites[1:]), 1):
        f = preparar_fold(df, variables, a, b, horizonte, ventana, objetivo)
        filas.append({'Fold': k, 'Train desde': f['fechas_train'][0],
                      'Train hasta': f['fechas_train'][-1],
                      'Val desde': f['fechas_val'][0], 'Val hasta': f['fechas_val'][-1],
                      'N train': len(f['y_train']), 'N val': len(f['y_val'])})
    return pd.DataFrame(filas), (df.index[inicio_test], df.index[-1])


def _semillas(config):
    return list(config.get('semillas') or [config['semilla']])


def _modelo(tipo, n, config, semilla):
    fijar_semilla(semilla)
    return crear_modelo(tipo=tipo, input_size=n,
                        hidden_size_1=config['hidden_size_1'],
                        hidden_size_2=config['hidden_size_2'],
                        dropout=config['dropout'],
                        residual=config.get('residual', False))


def _ajuste_fijo(tipo, f, config, epocas, semilla):
    # Reajuste con todos los datos elegibles, sin usar el bloque externo.
    modelo = _modelo(tipo, f['X_train'].shape[-1], config, semilla)
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo.to(dispositivo)
    ds = TensorDataset(torch.as_tensor(f['X_train'], dtype=torch.float32),
                       torch.as_tensor(f['y_train'], dtype=torch.float32))
    generador = torch.Generator().manual_seed(semilla)
    loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True,
                        generator=generador)
    criterio = nn.HuberLoss(delta=config['delta'])
    opt = torch.optim.Adam(modelo.parameters(), lr=config['lr'])
    for _ in range(epocas):
        modelo.train()
        for X, y in loader:
            opt.zero_grad()
            loss = criterio(modelo(X.to(dispositivo)), y.to(dispositivo).reshape(-1, 1))
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
    return modelo


# Nombre público: reentrenar un número fijo de épocas (usado en producción)
ajustar_epocas_fijas = _ajuste_fijo


def _evaluar_bloque(df, variables, inicio, fin, horizonte, ventana, modelos, config,
                    objetivo, devolver_predicciones=False):
    externo = preparar_fold(df, variables, inicio, fin, horizonte, ventana, objetivo)
    # Early stopping en una validación interna, anterior al origen externo.
    corte = inicio - horizonte + 1
    inicio_interno = int(corte * 0.90)
    interno = preparar_fold(df, variables, inicio_interno, corte, horizonte, ventana, objetivo)

    sy = externo['scaler_y']
    origen = externo['precio_origen_val']
    y = externo['precio_real_val']
    naive = origen
    mae_naive = calcular_metricas(y, naive)['MAE']

    filas = [{'Modelo': 'Ingenuo', **calcular_metricas(y, naive), 'Skill': 0.0,
              'DM estad.': np.nan, 'DM p-valor': np.nan, 'Mejor época': None}]
    predicciones = {'Fecha': externo['fechas_val'], 'Real': y, 'Ingenuo': naive}

    def fila(nombre, pred, epocas, alfa=np.nan):
        metricas = calcular_metricas(y, pred, origen)
        dm, p = test_diebold_mariano(y, pred, naive, horizonte)
        return {'Modelo': nombre, **metricas,
                'Skill': skill_vs_ingenuo(metricas['MAE'], mae_naive),
                'DM estad.': dm, 'DM p-valor': p,
                'Mejor época': float(np.mean(epocas)), 'Alfa': alfa}

    for tipo in modelos:
        preds_norm, preds_interno, epocas = [], [], []
        inicio_tiempo = time.perf_counter()
        for semilla in _semillas(config):
            modelo = _modelo(tipo, interno['X_train'].shape[-1], config, semilla)
            modelo_interno, hist = entrenar_modelo(
                modelo, interno['X_train'], interno['y_train'],
                interno['X_val'], interno['y_val'],
                epocas=config['epocas'], batch_size=config['batch_size'],
                lr=config['lr'], paciencia=config['paciencia'],
                semilla=semilla, verbose=config['verbose'])
            # Predicciones fuera de muestra en la validación interna (para elegir alfa)
            preds_interno.append(np.asarray(predecir_modelo(modelo_interno, interno['X_val'])))
            mejor = int(hist['mejor_epoca'])
            modelo = _ajuste_fijo(tipo, externo, config, mejor, semilla)
            preds_norm.append(np.asarray(predecir_modelo(modelo, externo['X_val'])))
            epocas.append(mejor)

        # Ensemble: media de las predicciones normalizadas de todas las semillas
        pred = reconstruir_precio(np.mean(preds_norm, axis=0), sy, origen, objetivo)
        filas.append({**fila(tipo, pred, epocas),
                      'Parámetros': contar_parametros(modelo),
                      # tiempo total (validación interna + reajuste, todas las semillas)
                      'Tiempo (s)': time.perf_counter() - inicio_tiempo})
        predicciones[tipo] = pred

        if config['contraccion']:
            # alfa elegido en la validación interna (anterior al fold): sin fuga
            pred_interno = reconstruir_precio(
                np.mean(preds_interno, axis=0), interno['scaler_y'],
                interno['precio_origen_val'], objetivo)
            alfa = elegir_alfa(interno['precio_real_val'], pred_interno,
                               interno['precio_origen_val'],
                               rejilla=np.linspace(0, config['alfa_max'],
                                                   int(round(config['alfa_max'] * 20)) + 1))
            pred_alfa = contraer_prediccion(pred, origen, alfa)
            filas.append(fila(f'{tipo} (α)', pred_alfa, epocas, alfa))
            predicciones[f'{tipo} (α)'] = pred_alfa

    if config['combinaciones'] and len(modelos) > 0:
        # Combinaciones con pesos IGUALES fijados a priori (Baumeister y Kilian):
        # no se ajusta ningún peso, así que no hay riesgo de sobreajuste.
        futuro = externo['futuro_origen_val'].copy()
        hay_futuro = np.isfinite(futuro).all()
        redes = np.mean([predicciones[t] for t in modelos], axis=0)
        predicciones['Futuro'] = futuro
        miembros = {'Comb. redes': [redes]}
        if hay_futuro:
            filas.append(fila('Futuro', futuro, [np.nan]))
            miembros |= {'Comb. ingenuo+futuro': [naive, futuro],
                         'Comb. redes+futuro': [redes, futuro],
                         'Comb. redes+ingenuo+futuro': [redes, naive, futuro]}
        for nombre, preds in miembros.items():
            pred_comb = np.mean(preds, axis=0)
            filas.append(fila(nombre, pred_comb, [np.nan]))
            predicciones[nombre] = pred_comb

    if devolver_predicciones:
        return filas, pd.DataFrame(predicciones).set_index('Fecha')
    return filas


def _config(hiperparametros):
    return dict(hidden_size_1=64, hidden_size_2=32, dropout=0.2, epocas=30,
                batch_size=32, lr=0.001, paciencia=5, semilla=42, semillas=None,
                delta=0.5, residual=False, verbose=False,
                contraccion=False, alfa_max=1.0,
                combinaciones=False) | (hiperparametros or {})


def evaluar_walk_forward(datos, horizontes=(5,), modelos=('RNN', 'LSTM', 'GRU'),
                         n_folds=5, ventana=60, hiperparametros=None, objetivo="nivel",
                         futuros=None):
    """
    Devuelve una fila por (horizonte, fold, modelo) con MAE, RMSE, sesgo,
    acierto direccional, skill vs ingenuo y test de Diebold-Mariano
    (estadístico < 0 y p < 0.05 -> el modelo bate al ingenuo con significación).

    hiperparametros['semillas'] = [0, 1, 2] -> ensemble de varias semillas.
    hiperparametros['contraccion'] = True -> añade una fila "<red> (α)" con la
    predicción contraída hacia el ingenuo (alfa elegido en la validación interna).
    hiperparametros['alfa_max'] = 2.0 -> permite también AMPLIFICAR la señal
    (alfa en [0, 2], paso 0.05). Por defecto alfa en [0, 1].
    hiperparametros['combinaciones'] = True -> añade el futuro del Brent como
    predicción y combinaciones con pesos iguales de redes, ingenuo y futuro.

    futuros (solo objetivo "retorno"): None, "wti" o "todos" -> añade variables
    de la curva de futuros (ver preprocessing.VARIABLES_FUTUROS_*).
    """
    df, variables, limites, _ = preparar_bloques(datos, n_folds, ventana, objetivo, futuros)
    config = _config(hiperparametros)
    etiqueta_futuros = futuros or 'ninguno'
    filas = []
    for h in horizontes:
        for k, (a, b) in enumerate(zip(limites[:-1], limites[1:]), 1):
            print(f'HORIZONTE {h} | FOLD {k}/{n_folds} | objetivo={objetivo} '
                  f'| futuros={etiqueta_futuros}', flush=True)
            for fila in _evaluar_bloque(df, variables, a, b, h, ventana, modelos,
                                        config, objetivo):
                filas.append({'Objetivo': objetivo, 'Futuros': etiqueta_futuros,
                              'Horizonte': h, 'Fold': k, **fila})
    return pd.DataFrame(filas)


def resumen_walk_forward(tabla):
    """Skill medio, folds ganados al ingenuo y MAE medio por modelo."""
    t = tabla[tabla['Modelo'] != 'Ingenuo']
    grupos = [c for c in ('Objetivo', 'Futuros', 'Horizonte', 'Modelo') if c in t.columns]
    return (t.groupby(grupos)
             .agg(MAE_medio=('MAE', 'mean'), Skill_medio=('Skill', 'mean'),
                  Folds_ganados=('Skill', lambda s: int((s > 0).sum())),
                  Folds=('Skill', 'size'),
                  Acierto_dir_medio=('Acierto dir. (%)', 'mean'),
                  Sesgo_medio=('Sesgo', 'mean'),
                  Alfa_medio=('Alfa', 'mean'))
             .round(3))


def evaluar_test_final(datos, horizonte, modelo, ventana=60, hiperparametros=None,
                       objetivo="nivel", devolver_predicciones=False, futuros=None):
    # Ejecutar únicamente después de congelar la selección de modelos e hiperparámetros.
    df, variables, _, inicio_test = preparar_bloques(datos, 5, ventana, objetivo, futuros)
    salida = _evaluar_bloque(df, variables, inicio_test, len(df), horizonte, ventana,
                             (modelo,), _config(hiperparametros), objetivo,
                             devolver_predicciones)
    if devolver_predicciones:
        filas, preds = salida
        return pd.DataFrame(filas).assign(Horizonte=horizonte, Objetivo=objetivo), preds
    return pd.DataFrame(salida).assign(Horizonte=horizonte, Objetivo=objetivo)
