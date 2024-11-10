# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='inicio'),
    path('predecir', views.predecir, name='predecir'),
]
