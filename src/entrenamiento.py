
import copy
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# ==========================================
# 1. REPRODUCIBILIDAD
# ==========================================

def fijar_semilla(semilla=42):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


# ==========================================
# 2. DISPOSITIVO
# ==========================================

def obtener_dispositivo():

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Dispositivo:", dispositivo)

    if dispositivo.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return dispositivo


# ==========================================
# 3. DATALOADERS
# ==========================================

def crear_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=32
):

    train_dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.as_tensor(X_val, dtype=torch.float32),
        torch.as_tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


# ==========================================
# 4. ENTRENAMIENTO DE UN MODELO
# ==========================================

def entrenar_modelo(
    modelo,
    X_train,
    y_train,
    X_val,
    y_val,
    epocas=30,
    batch_size=32,
    lr=0.001,
    paciencia=5,
    semilla=42
):

    fijar_semilla(semilla)

    dispositivo = obtener_dispositivo()
    modelo = modelo.to(dispositivo)

    train_loader, val_loader = crear_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size
    )

    criterio = nn.HuberLoss(delta=0.5)

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=lr
    )

    historial = {
        "train_loss": [],
        "val_loss": []
    }

    mejor_val_loss = float("inf")
    mejor_estado = None
    mejor_epoca = 0
    epocas_sin_mejora = 0

    inicio = time.perf_counter()

    for epoca in range(1, epocas + 1):

        # ==================================
        # ENTRENAMIENTO
        # ==================================

        modelo.train()
        perdida_train = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(dispositivo)
            y_batch = y_batch.to(dispositivo).reshape(-1, 1)

            optimizador.zero_grad()

            predicciones = modelo(X_batch)

            loss = criterio(predicciones, y_batch)

            loss.backward()

            # Control de gradientes explosivos
            torch.nn.utils.clip_grad_norm_(
                modelo.parameters(),
                max_norm=1.0
            )

            optimizador.step()

            perdida_train += loss.item() * len(X_batch)

        perdida_train /= len(train_loader.dataset)

        # ==================================
        # VALIDACION
        # ==================================

        modelo.eval()
        perdida_val = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(dispositivo)
                y_batch = y_batch.to(dispositivo).reshape(-1, 1)

                predicciones = modelo(X_batch)

                loss = criterio(predicciones, y_batch)

                perdida_val += loss.item() * len(X_batch)

        perdida_val /= len(val_loader.dataset)

        historial["train_loss"].append(perdida_train)
        historial["val_loss"].append(perdida_val)

        print(
            f"Época {epoca:02d}/{epocas} | "
            f"Train Loss: {perdida_train:.6f} | "
            f"Val Loss: {perdida_val:.6f}"
        )

        # ==================================
        # EARLY STOPPING
        # ==================================

        if perdida_val < mejor_val_loss - 1e-6:

            mejor_val_loss = perdida_val
            mejor_estado = copy.deepcopy(
                modelo.state_dict()
            )
            mejor_epoca = epoca
            epocas_sin_mejora = 0

        else:
            epocas_sin_mejora += 1

        if epocas_sin_mejora >= paciencia:

            print(
                f"Early stopping en la época {epoca}"
            )
            break

    tiempo = time.perf_counter() - inicio

    # Recuperar los mejores pesos
    modelo.load_state_dict(mejor_estado)

    historial["mejor_epoca"] = mejor_epoca
    historial["mejor_val_loss"] = mejor_val_loss
    historial["tiempo_segundos"] = tiempo

    print("\nENTRENAMIENTO FINALIZADO")
    print("Mejor época:", mejor_epoca)
    print("Mejor pérdida de validación:", round(mejor_val_loss, 6))
    print("Tiempo:", round(tiempo, 2), "segundos")

    return modelo, historial


# ==========================================
# 5. PREDICCIONES
# ==========================================

def predecir_modelo(
    modelo,
    X,
    batch_size=128
):

    dispositivo = next(modelo.parameters()).device

    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    predicciones = []

    modelo.eval()

    with torch.no_grad():

        for (X_batch,) in loader:

            X_batch = X_batch.to(dispositivo)

            salida = modelo(X_batch)

            predicciones.append(
                salida.cpu().numpy()
            )

    return np.concatenate(predicciones).ravel()


# ==========================================
# 6. GRAFICAS INDIVIDUALES
# ==========================================

def graficar_perdidas(historial, titulo="RNN"):

    epocas = np.arange(
        1,
        len(historial["train_loss"]) + 1
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        epocas,
        historial["train_loss"],
        marker="o",
        label="Entrenamiento"
    )

    plt.plot(
        epocas,
        historial["val_loss"],
        marker="o",
        label="Validación"
    )

    plt.axvline(
        historial["mejor_epoca"],
        color="gray",
        linestyle="--",
        label="Mejor época"
    )

    plt.title(f"Curvas de pérdida - {titulo}")
    plt.xlabel("Época")
    plt.ylabel("Pérdida normalizada")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. COMPARACION DE CURVAS
# ==========================================

def comparar_curvas(historiales):

    plt.figure(figsize=(12, 6))

    for nombre, historial in historiales.items():

        epocas = np.arange(
            1,
            len(historial["val_loss"]) + 1
        )

        plt.plot(
            epocas,
            historial["val_loss"],
            marker="o",
            label=nombre
        )

    plt.title("Comparación de pérdida de validación")
    plt.xlabel("Época")
    plt.ylabel("MSE normalizado")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
