
# BRENT. | Predicción del precio del petróleo con RNN, LSTM y GRU

**Proyecto de Aprendizaje Automático · Ingeniería Matemática · Universidad Alfonso X el Sabio (UAX)**

[**Ver aplicación web BRENT.**](https://caso02-petroleo.vercel.app) · [Repositorio en GitHub](https://github.com/claudiaalozano/caso02_petroleo)

## Descripción

Este proyecto estudia la capacidad de las redes neuronales recurrentes **RNN, LSTM y GRU** para predecir el precio del petróleo **Brent**.

Se comparan dos formulaciones del problema:

- **NIVEL:** predicción directa del precio futuro del Brent.
- **RETORNO:** predicción de sus retornos logarítmicos.

Ambas se analizan en tres horizontes temporales: **5, 63 y 252 días bursátiles**.

El objetivo no es únicamente reducir el error de las redes neuronales, sino comprobar si realmente mejoran un **modelo ingenuo** que predice que el precio futuro será igual al último precio conocido.

Además, se analizan la estabilidad temporal de los modelos, la combinación de predicciones y la incertidumbre asociada a sus estimaciones.

## Objetivos

- Explorar la evolución histórica del Brent y las variables del mercado disponibles.
- Preparar secuencias temporales para entrenar RNN, LSTM y GRU.
- Entrenar las tres arquitecturas para los horizontes de **5, 63 y 252 días**.
- Comparar las formulaciones **NIVEL** y **RETORNO**.
- Evaluar los modelos respetando la cronología mediante validación *Walk-Forward* y test final.
- Comparar el rendimiento de las redes con el modelo ingenuo.
- Estudiar combinaciones de predicciones e intervalos adaptados a la volatilidad.
- Publicar los resultados en una aplicación web interactiva.

## Datos y preprocesamiento

El análisis utiliza el dataset del proyecto, almacenado en:

`data/petroleo_dataset.csv`

Este contiene información histórica del Brent y otras variables del mercado.

En la configuración inicial se utilizan:

| Característica | Configuración |
|---|---|
| Variables predictoras | 13 |
| Longitud de la ventana temporal | 60 observaciones |
| Horizonte corto | 5 días bursátiles |
| Horizonte medio | 63 días bursátiles |
| Horizonte largo | 252 días bursátiles |

Las entradas de las redes neuronales tienen dimensiones:

$$
X \in \mathbb{R}^{N \times 60 \times 13}
$$

Donde:

- **N:** número de secuencias.
- **60:** observaciones anteriores.
- **13:** variables predictoras.

### Variables objetivo

Se estudian dos formulaciones:

**1. NIVEL**

El modelo estima directamente el precio futuro del Brent en USD/barril.

**2. RETORNO**

El modelo estima el retorno logarítmico:

$$
r_{t,h}=\ln\left(\frac{P_{t+h}}{P_t}\right)
$$

Posteriormente, el retorno predicho se transforma a precio:

$$
\hat{P}_{t+h}=P_t e^{\hat{r}_{t,h}}
$$

La separación entre entrenamiento, validación y test respeta el orden temporal para evitar utilizar información futura.

Los escaladores y transformadores que aprenden parámetros de los datos deben ajustarse únicamente con la información disponible durante cada etapa de entrenamiento.

## Modelos implementados

Se comparan tres arquitecturas recurrentes:

| Modelo | Características |
|---|---|
| **RNN** | Red recurrente simple que mantiene un estado oculto con información de observaciones anteriores. |
| **LSTM** | Incorpora mecanismos de memoria y puertas para aprender dependencias temporales. |
| **GRU** | Utiliza puertas de actualización y reinicio, con una arquitectura más compacta que LSTM. |

También se utiliza un **modelo ingenuo** como referencia.

### Modelo ingenuo

Predice que el precio futuro será igual al último precio conocido:

$$
\hat{P}_{t+h}=P_t
$$

Este modelo no requiere entrenamiento y constituye una referencia exigente para evaluar la utilidad de las redes neuronales.

## Entrenamiento de los modelos

### Entrenamiento inicial: tres horizontes

El entrenamiento inicial se realiza para los **tres horizontes temporales: 5, 63 y 252 días bursátiles**.

Para cada horizonte se entrenan independientemente:

- RNN.
- LSTM.
- GRU.

Por tanto, esta fase comprende **nueve entrenamientos iniciales**.

La variable objetivo utilizada en esta primera fase es **NIVEL**, es decir, la predicción directa del precio futuro del Brent.

### Configuración inicial

| Hiperparámetro | Valor |
|---|---|
| Variables de entrada | 13 |
| Longitud de secuencia | 60 |
| Primera capa oculta | 64 unidades |
| Segunda capa oculta | 32 unidades |
| Dropout | 0,2 |
| Learning rate | 0,001 |
| Batch size | 32 |
| Épocas máximas | 30 |
| Early stopping | Paciencia de 5 |
| Semilla | 42 |

Para cada modelo y horizonte se estudian las curvas de pérdida de entrenamiento y validación, así como sus métricas predictivas.

### Experimentos posteriores

Posteriormente se amplía el análisis para estudiar las formulaciones **NIVEL y RETORNO**.

En esta etapa también se utiliza una configuración de **32 y 16 unidades ocultas**, diferente de la configuración del entrenamiento inicial.

Los experimentos incluyen:

- Validación temporal *Walk-Forward*.
- Comparación entre arquitecturas.
- Análisis del modelo ingenuo.
- Evaluación sobre un test final.
- Comparación de diferentes horizontes de predicción.

## Validación temporal Walk-Forward

Para estudiar la capacidad de generalización se utiliza validación temporal mediante **cinco particiones** en el experimento de cinco días.

En cada partición:

1. El modelo se entrena con observaciones anteriores.
2. Se evalúa sobre un periodo posterior.
3. Se avanza temporalmente.
4. Se repite el proceso para analizar la estabilidad del rendimiento.

Este procedimiento permite evaluar los modelos respetando la estructura temporal de los datos.

Además, se reserva un conjunto de **test final** para comparar las distintas configuraciones.

## Métricas de evaluación

Para analizar el rendimiento predictivo se utilizan distintas métricas.

### MAE — Mean Absolute Error

Error absoluto medio:

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|P_i-\hat{P}_i|
$$

Indica el error promedio en USD/barril.

**Cuanto menor sea el MAE, mejor será el ajuste.**

### RMSE — Root Mean Squared Error

Penaliza especialmente los errores grandes.

### Skill frente al modelo ingenuo

Se define como:

$$
Skill=1-\frac{MAE_{modelo}}{MAE_{ingenuo}}
$$

Interpretación:

- **Skill positivo:** el modelo mejora al ingenuo.
- **Skill igual a cero:** ambos obtienen el mismo MAE.
- **Skill negativo:** el modelo tiene mayor error que el ingenuo.

También se estudian el sesgo, el acierto direccional y el contraste estadístico de Diebold-Mariano.

## Resultados del test final

Se evalúan las arquitecturas RNN, LSTM y GRU para los dos objetivos predictivos y los tres horizontes temporales.

Esto supone comparar **seis alternativas neuronales por horizonte**.

### Mejor red neuronal por horizonte

| Horizonte | Mejor red neuronal | MAE (USD/barril) | Skill |
|---|---|---:|---:|
| **5 días** | **RNN · RETORNO** | **3,669** | **−0,6 %** |
| **63 días** | **RNN · RETORNO** | **12,735** | **−12,8 %** |
| **252 días** | **RNN · NIVEL** | **16,967** | **−3,1 %** |

### Horizonte de 5 días

La **RNN de RETORNO** obtiene el menor MAE entre las redes neuronales, con **3,669 USD/barril**.

Sin embargo, el modelo ingenuo obtiene un MAE de **3,646 USD/barril**.

La diferencia es pequeña y, según el análisis realizado, los modelos de retorno no presentan diferencias estadísticamente significativas respecto al ingenuo al nivel del 5 % en el contraste de Diebold-Mariano.

En este horizonte, predecir retornos resulta considerablemente más competitivo que predecir directamente el nivel del precio.

### Horizonte de 63 días

La **RNN de RETORNO** vuelve a ser la mejor alternativa neuronal, con un MAE de **12,735 USD/barril**.

No obstante, su Skill es del **−12,8 %**, lo que significa que presenta un error superior al modelo ingenuo.

Los modelos tienen dificultades para anticipar cambios bruscos de tendencia y periodos de alta volatilidad.

### Horizonte de 252 días

La **RNN de NIVEL** obtiene el menor MAE entre las redes neuronales, con **16,967 USD/barril**.

Su Skill es del **−3,1 %**.

A diferencia de los horizontes anteriores, la formulación RETORNO no resulta la más competitiva entre las redes.

Esto muestra que la formulación del objetivo influye en el rendimiento y que su conveniencia depende del horizonte temporal.

### Conclusión del test final

**Ninguna de las redes seleccionadas consigue superar al modelo ingenuo en términos de MAE.**

Aunque RNN obtiene el menor error neuronal en los tres horizontes, no se puede concluir que sea universalmente superior a LSTM o GRU.

Además, la selección de las mejores redes se realiza de forma descriptiva sobre el propio test final, y no debe interpretarse como una selección prospectiva validada previamente.

## Comparación Predicho vs. Real

Se analiza la relación entre los precios reales y las predicciones generadas por las distintas redes.

En los gráficos de dispersión:

- El eje X representa el precio real.
- El eje Y representa el precio predicho.
- La diagonal **y = x** representa una predicción perfecta.
- Los puntos por encima de la diagonal indican sobreestimación.
- Los puntos por debajo indican infraestimación.

En el experimento específico de **NIVEL a 5 días**, GRU obtiene el menor MAE entre las tres redes, aproximadamente **4,44 USD/barril**.

Sin embargo, al incluir también RETORNO, la mejor alternativa neuronal de cinco días pasa a ser **RNN de RETORNO**.

Esta diferencia es importante para interpretar correctamente los resultados.

## Combinación de modelos

Además de estudiar individualmente las arquitecturas, se analizan estrategias de combinación de predicciones.

El objetivo es aprovechar información complementaria de diferentes modelos y reducir la dependencia de una sola arquitectura.

Se estudian combinaciones de redes recurrentes y referencias sencillas del mercado.

Estas estrategias se evalúan mediante validación temporal para comprobar si consiguen reducir el error.

No se presupone que una combinación vaya a mejorar necesariamente a todos los modelos individuales.

## Incertidumbre y volatilidad

Además de las predicciones puntuales, se construyen intervalos predictivos nominales del **50 % y del 80 %**.

Para adaptar los intervalos a la volatilidad del mercado, se utiliza el método **EWMA (Exponentially Weighted Moving Average)**.

La configuración incluye:

- Parámetro EWMA: **λ = 0,94**.
- Parámetro de reversión a la media: **φ = 0,99**.

El objetivo es que los intervalos reflejen los cambios en las condiciones del mercado.

### Calibración de intervalos

Para intervalos nominales del 80 %, se obtienen las siguientes coberturas observadas:

| Régimen de volatilidad | Cobertura |
|---|---:|
| Baja | 71,6 % |
| Media | 85,9 % |
| Alta | 82,4 % |

Los resultados muestran que la cobertura depende del régimen de volatilidad y que un intervalo nominal del 80 % no garantiza una cobertura futura exacta del 80 %.

## Prueba a ciegas

Se realiza una prueba a ciegas utilizando información disponible hasta el **18 de noviembre de 2026**.

El último precio spot conocido es de **92,94 USD/barril**.

Se generan predicciones para el 19 y el 20 de noviembre.

| Fecha | Predicción combinada | Futuro Brent observado | Diferencia absoluta |
|---|---:|---:|---:|
| 19/11/2026 | 92,77 USD | 91,98 USD | 0,79 USD |
| 20/11/2026 | 92,43 USD | 90,81 USD | 1,62 USD |

Ambas cotizaciones observadas quedaron dentro de los intervalos nominales del 80 %.

La cobertura observada fue de **2 de 2 valores (100 %)**.

Sin embargo, existen limitaciones importantes:

- Solo se dispone de dos observaciones.
- El valor del 20 de noviembre se consultó como provisional.
- Las predicciones proceden de Brent spot.
- Las observaciones posteriores corresponden a futuros Brent.

Por ello, esta comparación es ilustrativa y no demuestra una cobertura estadística real del 100 % ni constituye una evaluación sobre el mismo instrumento financiero.

## Aplicación web BRENT.

### [Acceder a la aplicación interactiva](https://caso02-petroleo.vercel.app)

Se ha desarrollado una plataforma web utilizando **React, TypeScript y Vite**, desplegada mediante **Vercel**.

La aplicación permite:

- Consultar la evolución histórica del Brent.
- Explorar predicciones para los distintos horizontes.
- Seleccionar modelos y objetivos predictivos.
- Comparar métricas de evaluación.
- Visualizar gráficos de Predicho vs. Real.
- Analizar la fiabilidad y validación temporal.
- Estudiar el error absoluto acumulado.
- Consultar la calibración de intervalos.
- Visualizar la prueba a ciegas.
- Explorar acontecimientos históricos relevantes del mercado petrolero.

La aplicación presenta resultados calculados previamente en Python. No entrena los modelos directamente desde el navegador.

## Estructura del repositorio

```text
caso02_petroleo/
│
├── analisis_petroleo.ipynb
│   └── Notebook principal de análisis y experimentos
│
├── data/
│   └── Dataset histórico del proyecto
│
├── src/
│   └── Preprocesamiento, modelos y evaluación
│
├── scripts/
│   └── Scripts auxiliares y exportación de resultados
│
├── resultados/
│   └── Entrenamientos, métricas, predicciones y figuras
│
├── app/
│   ├── web/
│   │   └── Componentes React y estilos
│   └── public/datos/
│       └── Archivos JSON utilizados por la web
│
└── README.md
```

La estructura y los nombres de algunos archivos pueden variar según los experimentos ejecutados.

## Ejecución del proyecto

### Entorno Python

El proyecto utiliza Python y librerías como:

- PyTorch.
- NumPy.
- pandas.
- scikit-learn.
- Matplotlib.
- Jupyter.

Si el entorno Conda del proyecto ya está creado:

```bash
conda activate petroleo_gpu
```

Para abrir el notebook desde la raíz del repositorio:

```bash
jupyter lab analisis_petroleo.ipynb
```

En otros equipos será necesario configurar un entorno equivalente e instalar las dependencias.

### Aplicación web

Es necesario tener instalados Node.js y npm.

Desde la raíz del repositorio:

```bash
cd app
npm install
npm run dev
```

Para comprobar la compilación:

```bash
npm run build
```

Los datos mostrados por la aplicación se almacenan en archivos JSON publicados dentro de `app/public/datos/`.

## Tecnologías utilizadas

**Machine Learning y análisis de datos**

Python · PyTorch · NumPy · pandas · scikit-learn · Matplotlib · Jupyter

**Aplicación web**

React · TypeScript · Vite · Vercel

**Control de versiones**

Git · GitHub

## Conclusiones generales

Los resultados del proyecto muestran que las redes neuronales recurrentes pueden aprender relaciones presentes en las series históricas del petróleo Brent.

Sin embargo, **una mayor complejidad arquitectónica no garantiza una mayor capacidad predictiva**.

La RNN obtiene el menor MAE entre las alternativas neuronales seleccionadas en cada horizonte del test final, pero ninguna de ellas supera al modelo ingenuo.

También se observa que:

- La formulación RETORNO resulta más competitiva en los horizontes de 5 y 63 días.
- A 252 días, NIVEL presenta el mejor resultado neuronal.
- Los cambios bruscos del mercado dificultan la generalización.
- La combinación de modelos no garantiza por sí misma una mejora.
- Cuantificar la incertidumbre es tan importante como obtener una predicción puntual.

**La conclusión principal es que la calidad de un modelo financiero no debe medirse únicamente por su complejidad, sino por su capacidad de generalizar, superar referencias sencillas y comunicar adecuadamente la incertidumbre.**

---

**Autora:** Claudia Lozano  
**Grado:** Ingeniería Matemática — UAX  
**Asignatura:** Aprendizaje Automático  
**Aplicación web:** https://caso02-petroleo.vercel.app
