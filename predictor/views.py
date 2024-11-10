from django.shortcuts import render
from tensorflow.keras.models import load_model
import os
import numpy as np
from django.conf import settings

# Vista para la página principal
def index(request):
    return render(request, 'paginas/inicio.html')

################################################################ PREDECIR

# Ruta del modelo preentrenado
model_path = os.path.join(settings.BASE_DIR, 'predictor', 'modelo', 'modelo', 'modelo-casa-predictorRN-V6.h5')

# Intentar cargar el modelo solo si el archivo existe
try:
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        raise FileNotFoundError(f"No se pudo encontrar el archivo del modelo en: {model_path}")
except (FileNotFoundError, IOError) as e:
    model = None
    error_message = str(e)
    print(f"Error al cargar el modelo: {error_message}")  # Registra el error si es necesario

# Vista para predecir el precio de la casa
def predecir(request):
    if request.method == 'POST':
        if model is None:
            # Si el modelo no se ha cargado, mostrar un mensaje de error
            return render(request, 'paginas/predecir.html', {'error': 'El modelo no está disponible en este momento. Inténtalo más tarde.'})
        
        try:
            # Obtener los datos del formulario
            longitude = float(request.POST.get('longitude'))
            latitude = float(request.POST.get('latitude'))
            housing_median_age = float(request.POST.get('housing_median_age'))
            total_rooms = float(request.POST.get('total_rooms'))
            total_bedrooms = float(request.POST.get('total_bedrooms'))
            population = float(request.POST.get('population'))
            households = float(request.POST.get('households'))
            median_income = float(request.POST.get('median_income'))
        except ValueError:
            return render(request, 'paginas/predecir.html', {'error': 'Por favor, ingrese valores válidos para todas las características.'})

        # Preparar los datos para la predicción
        features = np.array([[longitude, latitude, housing_median_age, total_rooms,
                              total_bedrooms, population, households, median_income]])

        # Realizar la predicción
        prediction = model.predict(features)[0]*10000

        # Mostrar el resultado en la página
        return render(request, 'paginas/predecir.html', {'resultado': prediction})

    # Si el método no es POST o si algo falla, mostrar la página sin resultado
    return render(request, 'paginas/predecir.html')
