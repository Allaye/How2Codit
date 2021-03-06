from django.urls import path
from progress import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signin/', views.signin, name='signin'),
    path('logout/', views.signout, name='logout'),
    path('register/', views.registration, name='register'),
    path('profile/', views.profile, name='profile'),
    path('upload/', views.simple_upload, name='upload'),
    # path('home/', views.home, name='home')

    # path('progress/', views.progress, name='progress')
]