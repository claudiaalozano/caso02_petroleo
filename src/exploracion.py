
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CARGA DE DATOS
# ==========================================

def cargar_datos(ruta):
    """Carga el dataset y ordena los registros cronológicamente."""

    datos = pd.read_csv(ruta, parse_dates=["Date"])
    datos = datos.sort_values("Date")
    datos = datos.set_index("Date")

    return datos


# ==========================================
# 2. INFORMACION GENERAL
# ==========================================

def resumen_datos(datos):
    """Muestra dimensiones, fechas, variables y tipos."""

    print("DIMENSIONES:", datos.shape)

    print("\nPERIODO:")
    print("Desde:", datos.index.min())
    print("Hasta:", datos.index.max())

    print("\nVARIABLES:")
    print(datos.columns.tolist())

    print("\nTIPOS DE DATOS:")
    print(datos.dtypes)

    print("\nESTADISTICAS DESCRIPTIVAS:")
    return datos.describe().T


# ==========================================
# 3. VALORES FALTANTES
# ==========================================

def analizar_nulos(datos):
    """Calcula la cantidad y porcentaje de nulos."""

    nulos = pd.DataFrame({
        "Nulos": datos.isnull().sum(),
        "Porcentaje (%)": datos.isnull().mean() * 100
    })

    return nulos.sort_values(
        "Porcentaje (%)", ascending=False
    )


def graficar_nulos(datos):
    """Representa el porcentaje de nulos por variable."""

    porcentajes = datos.isnull().mean() * 100
    porcentajes = porcentajes.sort_values(ascending=False)

    plt.figure(figsize=(12, 5))

    sns.barplot(
        x=porcentajes.index,
        y=porcentajes.values,
        color="steelblue"
    )

    plt.title("Porcentaje de valores faltantes")
    plt.ylabel("Porcentaje (%)")
    plt.xlabel("Variables")
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. EVOLUCION TEMPORAL DEL BRENT
# ==========================================

def graficar_brent(datos):
    """Representa la evolución histórica del Brent."""

    plt.figure(figsize=(14, 6))

    plt.plot(
        datos.index,
        datos["brent"],
        color="darkblue",
        linewidth=1
    )

    plt.title("Evolución histórica del precio del Brent")
    plt.xlabel("Fecha")
    plt.ylabel("Precio Brent ($/barril)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. DISTRIBUCION DEL BRENT
# ==========================================

def distribucion_brent(datos):
    """Muestra la distribución de los precios del Brent."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(
        datos["brent"].dropna(),
        bins=50,
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title("Distribución del precio Brent")
    axes[0].set_xlabel("Precio ($/barril)")

    sns.boxplot(
        y=datos["brent"].dropna(),
        ax=axes[1]
    )

    axes[1].set_title("Diagrama de caja del Brent")
    axes[1].set_ylabel("Precio ($/barril)")

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. VALORES EXTREMOS
# ==========================================

def valores_extremos(datos):
    """Obtiene los diez precios más altos y más bajos."""

    mayores = datos["brent"].nlargest(10)
    menores = datos["brent"].nsmallest(10)

    return mayores, menores


# ==========================================
# 7. CORRELACIONES
# ==========================================

def matriz_correlacion(datos):
    """Calcula y representa correlaciones de Pearson."""

    correlaciones = datos.select_dtypes(
        include=np.number
    ).corr()

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        correlaciones,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f"
    )

    plt.title("Matriz de correlación entre variables")
    plt.tight_layout()
    plt.show()

    return correlaciones


# ==========================================
# 8. COMPARACION BRENT Y WTI
# ==========================================

def comparar_brent_wti(datos):
    """Compara las dos principales referencias del petróleo."""

    plt.figure(figsize=(14, 6))

    plt.plot(
        datos.index,
        datos["brent"],
        label="Brent",
        linewidth=1
    )

    plt.plot(
        datos.index,
        datos["wti"],
        label="WTI",
        linewidth=1,
        alpha=0.7
    )

    plt.title("Comparación histórica Brent vs WTI")
    plt.xlabel("Fecha")
    plt.ylabel("Precio ($/barril)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 9. DISPONIBILIDAD TEMPORAL
# ==========================================

def disponibilidad_variables(datos):
    """Muestra el primer y último dato disponible de cada variable."""

    resumen = pd.DataFrame({
        "Primer dato": datos.apply(
            lambda columna: columna.first_valid_index()
        ),
        "Último dato": datos.apply(
            lambda columna: columna.last_valid_index()
        ),
        "Nulos": datos.isnull().sum(),
        "Porcentaje nulos": (
            datos.isnull().mean() * 100
        ).round(2)
    })

    return resumen


# ==========================================
# 10. VARIACIONES DIARIAS DEL BRENT
# ==========================================

def analizar_variaciones_brent(datos):
    """Analiza las variaciones porcentuales diarias del Brent."""

    variaciones = datos["brent"].pct_change() * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Evolución temporal
    axes[0].plot(
        variaciones.index,
        variaciones,
        color="darkblue",
        linewidth=0.7
    )

    axes[0].set_title("Variaciones porcentuales diarias del Brent")
    axes[0].set_xlabel("Fecha")
    axes[0].set_ylabel("Variación diaria (%)")
    axes[0].grid(alpha=0.3)

    # Distribución
    sns.histplot(
        variaciones.dropna(),
        bins=100,
        kde=True,
        ax=axes[1]
    )

    axes[1].set_title("Distribución de las variaciones diarias")
    axes[1].set_xlabel("Variación diaria (%)")

    plt.tight_layout()
    plt.show()

    print("ESTADÍSTICAS DE LAS VARIACIONES:")
    print(variaciones.describe())

    return variaciones