from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def inicio(request):
    return render(request, 'paginas/inicio.html')


######################################################### PREDECIR
from django.shortcuts import render
import pickle
import os
import numpy as np
from django.conf import settings

# Obtener la ruta base del proyecto (BASE_DIR)
model_path = os.path.join(settings.BASE_DIR, 'predictor', 'modelo', 'modelo-casa-predictor.pkl')

# Verifica si el archivo existe antes de cargarlo
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    raise FileNotFoundError(f"No se pudo encontrar el archivo del modelo en: {model_path}")


def predecir(request):
    if request.method == 'POST':
        # Obtener los datos del formulario
        longitude = float(request.POST.get('longitude'))
        latitude = float(request.POST.get('latitude'))
        housing_median_age = float(request.POST.get('housing_median_age'))
        total_rooms = float(request.POST.get('total_rooms'))
        total_bedrooms = float(request.POST.get('total_bedrooms'))
        population = float(request.POST.get('population'))
        households = float(request.POST.get('households'))
        median_income = float(request.POST.get('median_income'))

        # Preparar los datos para el modelo
        features = np.array([[longitude, latitude, housing_median_age, total_rooms,
                              total_bedrooms, population, households, median_income]])
        prediction = model.predict(features)[0]
        prediction = prediction * 10000

        return render(request, 'paginas/predecir.html', {'resultado': prediction})

    return render(request, 'paginas/predecir.html')
 