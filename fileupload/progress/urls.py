from django.urls import path
from progress import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signin/', views.signin, name='signin'),
    path('logout/', views.logout, name='logout'),
    path('register/', views.registration, name='register'),
    path('profile/', views.profile, name='profile')

]