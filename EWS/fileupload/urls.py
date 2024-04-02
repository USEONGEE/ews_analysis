from django.urls import path

from . import views

urlpatterns = [
  path('analyze/', views.analysis, name="analysis"),
  path('analyze/v2/', views.analysisV2, name="analysis"),
]