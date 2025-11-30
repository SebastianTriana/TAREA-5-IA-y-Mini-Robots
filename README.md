# TAREA 5 – Inteligencia Artificial y Mini Robots

### Sebastián Triana – Juan Diego Camacho

### Universidad Nacional de Colombia – 2025-2

Este repositorio contiene el desarrollo completo de los tres puntos solicitados en la **Tarea #5** del curso *Inteligencia Artificial y Mini Robots*.
Incluye implementación de redes neuronales básicas, un modelo de clasificación usando **TensorFlow** con el dataset **Fashion-MNIST**, y un ejercicio con un dataset externo donde se entrena una red neuronal y se analizan sus pesos aprendidos.

---

# Contenido del repositorio

El repositorio está organizado en tres puntos principales, cada uno con su respectivo script y carpeta de resultados cuando aplica.

---

# Punto 1 — Redes neuronales para NAND y XOR

En este punto se entrenan **dos redes neuronales** (ambas con dos capas ocultas) para aprender las funciones lógicas:

* **NAND**
* **XOR**

Cada modelo:

* Se entrena con *backpropagation*.
* Evalúa las cuatro combinaciones posibles de entrada.
* Imprime los **pesos aprendidos** en cada capa.
* Guarda las predicciones y resultados.

**Archivos incluidos:**

* `punto1_NAND_XOR.py`
* `resultados/nand_predicciones.txt`
* `resultados/xor_predicciones.txt`

---

# Punto 2 — Clasificación con TensorFlow usando Fashion-MNIST

Se entrena una red neuronal sencilla que clasifica imágenes del dataset **Fashion-MNIST**.
El modelo utiliza las siguientes capas:

* `Flatten`
* `Dense` (capa oculta con activación ReLU)
* `Dense` con `softmax` (10 clases)

El programa genera:

* Accuracy de entrenamiento y prueba
* Matriz de confusión
* Ejemplos bien y mal clasificados

**Archivo relevante:**
`punto2_fashion_mnist.py`

---

# Punto 3 — Dataset externo: Iris

Para este punto se seleccionó el dataset **Iris** de *scikit-learn*.
El pipeline incluye:

1. Visualización de características y clases.
2. One-hot encoding del rótulo usando `OneHotEncoder(sparse_output=False)`.
3. Entrenamiento de una red neuronal totalmente conectada.
4. Evaluación del desempeño y matriz de confusión.
5. **Extracción y análisis de los pesos aprendidos** para interpretar la importancia de las features.

**Archivo relevante:**
`punto3_dataset_externo.py`

---

# Requisitos

Instalar dependencias con:

```
pip install numpy tensorflow scikit-learn matplotlib
```

---

# Cómo ejecutar

## Punto 1: NAND y XOR

```
python punto1_NAND_XOR.py
```

## Punto 2: Fashion MNIST

```
python punto2_fashion_mnist.py
```

## Punto 3: Dataset externo (Iris)

```
python punto3_dataset_externo.py
```

---

# Autores

**Sebastián Triana**
**Juan Diego Camacho**

Universidad Nacional de Colombia
Facultad de Ingeniería – 2025-2
