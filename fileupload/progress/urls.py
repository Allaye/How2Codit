from django.urls import path
from progress import views

urlpatterns = [
    path('/', views.index, name='index'),
    path('signin/', views.index, name='signin'),

]