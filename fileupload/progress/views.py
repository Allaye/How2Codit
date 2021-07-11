from time import sleep
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required 
from django.contrib.auth import authenticate, get_user_model, login, logout
from progress.forms import UserForm, UserDetailsUpdate, UserLogin
from progress.models import UserProfile



# Create your views here.
def index(request):
    # if request.method == "POST":
    #     form = UserForm(request.POST)
    #     if form.is_valid():
    #         user = form.save()
    #         user.set_password(user.password)
    #         user.save()
    #         return redirect('profile')
    # userform = UserForm()
    user = request.user
    return render(request, 'index.html', {'form': user})


def signin(request):
    form = UserLogin(request.POST)
    if form.is_valid():
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        user = authenticate(username=username, password=password)
        if user:
            login(request, user)
            return redirect('index')
        else:
            return redirect('signin')
    return render(request, 'login.html', {'form': form})

@login_required()
def logout(request):
    logout(request)
    return redirect('index')


@login_required()
def profile(request):
    if request.method == 'POST':
        user = UserProfile.objects.get(user=request.user)
        form = UserDetailsUpdate(request.POST, instance=user)
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




# def myprofile(request):
#     if request.method == 'POST':
        
#         form = UserProfileForm(request.POST, instance=request.user)
#         if form.is_valid():
#             form.save()
#             user_profile = UserProfile.objects.get(user=request.user)
#     if request.method == 'GET':
#         user_profile = UserProfile.objects.get(user=request.user)
#         form = UserProfileForm(instance=user_profile)

# def myprofile(request):
#     if request.method == 'POST':
#         user_profile = UserProfile.objects.get(user=request.user)
#         form = UserProfileForm(request.POST, instance=user_profile)
