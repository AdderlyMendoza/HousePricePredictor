# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('inicio', views.inicio, name='inicio'),
    path('predecir', views.predecir, name='predecir'),

]
