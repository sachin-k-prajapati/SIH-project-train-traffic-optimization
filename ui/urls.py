from django.urls import path
from . import views

app_name = 'ui'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('section-controller/', views.section_controller, name='section_controller'),
    path('ultimate/', views.ultimate_dashboard, name='ultimate_dashboard'),
    path('recommendations/', views.recommendations_view, name='recommendations'),
    path('kpis/', views.kpis_view, name='kpis'),
    path('api/recommendations/', views.api_recommendations, name='api_recommendations'),
    path('api/kpis/', views.api_kpis, name='api_kpis'),
]