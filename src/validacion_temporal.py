"""Walk-forward con ventana expansiva y test reservado para Brent."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import (
    preparar_variables, obtener_limites, normalizar_datos,
    crear_secuencias, VARIABLES_MODELO,
)
from src.modelos import crear_modelo
from src.entrenamiento import fijar_semilla, entrenar_modelo, predecir_modelo
from src.evaluacion import calcular_metricas


def preparar_bloques(datos, n_folds=5, ventana=60):
    df = preparar_variables(datos)
    if not df.index.is_unique:
        raise ValueError('Hay fechas duplicadas')
    inicio_val, inicio_test = obtener_limites(df)
    if n_folds < 2 or inicio_test - inicio_val < n_folds:
        raise ValueError('Número de folds no válido')
    limites = np.linspace(inicio_val, inicio_test, n_folds + 1, dtype=int)
    return df, limites, inicio_test


def preparar_fold(df, inicio, fin, horizonte, ventana=60):
    # Los objetivos de entrenamiento deben conocerse al emitir la primera predicción.
    corte = inicio - horizonte + 1
    if not (ventana + horizonte < corte <= inicio < fin <= len(df)):
        raise ValueError('No hay suficiente histórico para este bloque')
    Xs, ys, sx, sy = normalizar_datos(df.iloc[:fin], corte)
    X, y, idx = crear_secuencias(Xs, ys, ventana, horizonte)
    train = idx < corte
    val = (idx >= inicio) & (idx < fin)
    assert train.any() and val.any()
    assert idx[train].max() <= idx[val].min() - horizonte
    return dict(X_train=X[train], y_train=y[train], X_val=X[val], y_val=y[val],
                fechas_train=df.index[idx[train]], fechas_val=df.index[idx[val]],
                fecha_origen=df.index[inicio-horizonte], scaler_X=sx, scaler_y=sy)


def tabla_folds(datos, horizonte=5, n_folds=5, ventana=60):
    df, limites, inicio_test = preparar_bloques(datos, n_folds, ventana)
    filas = []
    for k, (a,b) in enumerate(zip(limites[:-1], limites[1:]), 1):
        f = preparar_fold(df, a, b, horizonte, ventana)
        filas.append({'Fold': k, 'Train desde': f['fechas_train'][0],
                      'Train hasta': f['fechas_train'][-1],
                      'Val desde': f['fechas_val'][0], 'Val hasta': f['fechas_val'][-1],
                      'N train': len(f['y_train']), 'N val': len(f['y_val'])})
    return pd.DataFrame(filas), (df.index[inicio_test], df.index[-1])


def _modelo(tipo, n, config):
    fijar_semilla(config['semilla'])
    return crear_modelo(tipo=tipo, input_size=n,
                        hidden_size_1=config['hidden_size_1'],
                        hidden_size_2=config['hidden_size_2'],
                        dropout=config['dropout'])


def _ajuste_fijo(tipo, f, config, epocas):
    # Reajuste con todos los datos elegibles, sin usar el bloque externo.
    modelo = _modelo(tipo, f['X_train'].shape[-1], config)
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo.to(dispositivo)
    ds = TensorDataset(torch.as_tensor(f['X_train'], dtype=torch.float32),
                       torch.as_tensor(f['y_train'], dtype=torch.float32))
    loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True)
    criterio = nn.HuberLoss(delta=0.5)
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


def _evaluar_bloque(df, inicio, fin, horizonte, ventana, modelos, config):
    externo = preparar_fold(df, inicio, fin, horizonte, ventana)
    # Early stopping en una validación interna, anterior al origen externo.
    corte = inicio - horizonte + 1
    inicio_interno = int(corte * 0.90)
    interno = preparar_fold(df, inicio_interno, corte, horizonte, ventana)
    sy = externo['scaler_y']; sx = externo['scaler_X']
    y = sy.inverse_transform(externo['y_val'].reshape(-1, 1)).ravel()
    naive = externo['X_val'][:, -1, 0] * sx.scale_[0] + sx.mean_[0]
    filas = [{'Modelo': 'Ingenuo', **calcular_metricas(y, naive), 'Mejor época': None}]
    for tipo in modelos:
        modelo = _modelo(tipo, interno['X_train'].shape[-1], config)
        _, hist = entrenar_modelo(modelo, interno['X_train'], interno['y_train'],
                                   interno['X_val'], interno['y_val'],
                                   epocas=config['epocas'], batch_size=config['batch_size'],
                                   lr=config['lr'], paciencia=config['paciencia'],
                                   semilla=config['semilla'])
        mejor = int(hist['mejor_epoca'])
        modelo = _ajuste_fijo(tipo, externo, config, mejor)
        pred = np.asarray(predecir_modelo(modelo, externo['X_val'])).reshape(-1, 1)
        pred = sy.inverse_transform(pred).ravel()
        filas.append({'Modelo': tipo, **calcular_metricas(y, pred), 'Mejor época': mejor})
    return filas


def _config(hiperparametros):
    return dict(hidden_size_1=64, hidden_size_2=32, dropout=0.2, epocas=30,
                batch_size=32, lr=0.001, paciencia=5, semilla=42) | (hiperparametros or {})


def evaluar_walk_forward(datos, horizontes=(5,), modelos=('RNN','LSTM','GRU'),
                          n_folds=5, ventana=60, hiperparametros=None):
    df, limites, _ = preparar_bloques(datos, n_folds, ventana)
    config = _config(hiperparametros)
    filas = []
    for h in horizontes:
        for k, (a,b) in enumerate(zip(limites[:-1], limites[1:]), 1):
            print(f'\nHORIZONTE {h} | FOLD {k}/{n_folds}')
            for fila in _evaluar_bloque(df, a, b, h, ventana, modelos, config):
                filas.append({'Horizonte': h, 'Fold': k, **fila})
    return pd.DataFrame(filas)


def evaluar_test_final(datos, horizonte, modelo, ventana=60, hiperparametros=None):
    # Ejecutar únicamente después de congelar la selección de modelos e hiperparámetros.
    df, _, inicio_test = preparar_bloques(datos, 5, ventana)
    filas = _evaluar_bloque(df, inicio_test, len(df), horizonte, ventana,
                            (modelo,), _config(hiperparametros))
    return pd.DataFrame(filas).assign(Horizonte=horizonte)
