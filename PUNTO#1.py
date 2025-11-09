# Ejercicio #1
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(42)
np.random.seed(42)

# Datos (entradas 2 bits)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)

# Targets
# NAND: 1 except cuando ambos son 1
y_nand = np.array([[1],[1],[1],[0]], dtype=np.float32)
# XOR: 1 si bits diferentes
y_xor  = np.array([[0],[1],[1],[0]], dtype=np.float32)

def build_model():
    model = Sequential([
        Dense(8, activation='relu', input_shape=(2,)),   # capa oculta 1
        Dense(4, activation='relu'),                     # capa oculta 2
        Dense(1, activation='sigmoid')                   # salida binaria
    ])
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Entrenamiento del NAND
model_nand = build_model()
model_nand.fit(X, y_nand, epochs=200, verbose=0)
print("NAND - predicciones:")
print(np.round(model_nand.predict(X).flatten(), 3))
print("NAND - valores reales:", y_nand.flatten())

# Entrenamiento del XOR
model_xor = build_model()
model_xor.fit(X, y_xor, epochs=400, verbose=0)
print("\nXOR - predicciones:")
print(np.round(model_xor.predict(X).flatten(), 3))
print("XOR - valores reales:", y_xor.flatten())

# Mostrar pesos de la primera capa (ejemplo)
print("\nPesos primera capa (NAND):")
for w,b in zip(model_nand.layers[0].get_weights()[0].T, model_nand.layers[0].get_weights()[1]):
    print("w:", np.round(w,3), "b:", np.round(b,3))
