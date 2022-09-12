from django.urls import path
from . import views

urlpatterns = [
    #path('', views.apiOverview, name='api-overview'),
    path('recommend', views.predict, name='recommend'),
    path('classify', views.classify, name='classify'),
]