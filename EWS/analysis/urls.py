from django.urls import path

from . import views

urlpatterns = [
  path('', views.analysis, name="analysis"),
  path('v2/', views.analysisV2, name="analysis"),
]