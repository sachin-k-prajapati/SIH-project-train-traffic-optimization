from django.urls import path
from api import views

urlpatterns = [
    path('train_events/', views.train_events, name='train_events'),
    path('disruptions/', views.disruptions, name='disruptions'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('decisions/override/', views.decision_override, name='decision_override'),
    path('kpis/', views.kpis, name='kpis'),
    path('sections/', views.SectionList.as_view(), name='section_list'),
    path('trains/', views.TrainList.as_view(), name='train_list'),
    path('decisions/', views.DecisionList.as_view(), name='decision_list'),
]