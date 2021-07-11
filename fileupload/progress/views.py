from time import sleep
from django.shortcuts import redirect, render
from progress.forms import UserForm, UserDetailsUpdate, UserLogin
from progress.models import UserProfile



# Create your views here.
def index(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.set_password(user.password)
            user.save()
            return redirect('profile')
    userform = UserForm()
    return render(request, 'index.html', {'form': userform})


def signin(request):
    return render(request, 'login.html')

def logout(request):
    return redirect(index)

def profile(request):
    if request.method == 'POST':
        form = UserDetailsUpdate(request.POST)
        if form.is_valid():
            form.save()
            return redirect('profile')
    form = UserDetailsUpdate()
    return render(request, 'profile.html', {'form': form})

def registration(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.set_password(user.password)
            user.save()
            return redirect('login')
    userform = UserForm()
    return render(request, 'register.html', {'form': userform})
