from time import sleep
from django.shortcuts import render
from progress.forms import UserForm
from progress.models import UserProfile



# Create your views here.
def index(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            return render(request, 'index.html')
    form = UserForm()
    return render(request, 'index.html', {'form': form})


def signin(request):
    return render(request, 'login.html')

def profile(request):
    return render(request, 'profile.html')

def registration(request):
    return render(request, 'register.html')