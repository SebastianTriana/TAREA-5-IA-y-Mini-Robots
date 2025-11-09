# Ejercicio 3: Ejemplo con Iris (análisis + red neuronal simple)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import models, layers

# Cargar y preparar
iris = load_iris()
X = iris['data']        # features: sepal length, sepal width, petal length, petal width
y = iris['target']      # labels: 0,1,2 (3 clases)
feature_names = iris['feature_names']
target_names = iris['target_names']

# Exploración rápida
print("Features:", feature_names)
print("Clases:", target_names)
print("Tamaño dataset:", X.shape)

# Normalizar y one-hot
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
ohe = OneHotEncoder(sparse=False)
y_onehot = ohe.fit_transform(y.reshape(-1,1))

# Split
Xtr, Xte, ytr, yte = train_test_split(Xs, y_onehot, test_size=0.2, random_state=42)

# Diseñar red (clasificador)
model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(Xtr.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(Xtr, ytr, epochs=60, batch_size=8, verbose=0)

loss, acc = model.evaluate(Xte, yte, verbose=0)
print(f"Test accuracy Iris: {acc:.4f}")

# Mostrar pesos de la primera capa (ejemplo)
W, b = model.layers[0].get_weights()
print("\nPesos (capa 1) shape:", W.shape)
print("Bias (capa 1):", np.round(b, 3))
print("Pesos (capa 1) (por feature):")
for i, fname in enumerate(feature_names):
    print(f" {fname}: {np.round(W[i,:], 3)}")

# Ejemplo de predicción y uso de pesos aprendidos:
sample = Xte[0:5]
preds = model.predict(sample)
print("\nEjemplo - predicciones (probabilidades):")
print(np.round(preds,3))
print("Clases predichas:", np.argmax(preds, axis=1))
print("Clases reales   :", np.argmax(yte[0:5], axis=1))
