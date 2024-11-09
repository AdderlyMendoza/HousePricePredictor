import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar el modelo entrenado desde el archivo .pkl
with open('C:\laragon\www\Predictor de Precios de Casas Django\HousePricePredictor\predictor\modelo\modelo-casa-predictor.pkl', 'rb') as file:
    model = pickle.load(file)

print("Modelo cargado exitosamente")

# Paso 2: Cargar el dataset de California
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# Paso 3: Dividir en características (X) y variable objetivo (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Paso 4: Dividir los datos en entrenamiento y prueba (mismo que antes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 5: Realizar predicciones usando el modelo cargado
y_pred = model.predict(X_test)

# Paso 6: Calcular métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMétricas de evaluación:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Paso 7: Visualización de Resultados
# Comparar los valores predichos con los valores reales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, 5], [0, 5], '--r', linewidth=2, color='red')  # Línea de referencia
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Comparación entre Valores Reales y Predichos")
plt.grid(True)
plt.show()

# Histograma de errores de predicción
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, bins=30, kde=True)
plt.xlabel("Error de Predicción")
plt.ylabel("Frecuencia")
plt.title("Distribución de los Errores de Predicción")
plt.grid(True)
plt.show()
