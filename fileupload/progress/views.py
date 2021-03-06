import io
from time import sleep
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required 
from django.contrib.auth import authenticate, get_user_model, login, logout
from progress.forms import UserForm, UserDetailsUpdate, UserLogin
from progress.models import Profile
from progress.tasks import process_download, read_chunk, process_download_with_progress
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.images import ImageFile
import pickle


import sys, os, math
def simple_upload(request):
    if request.method == 'POST' and request.FILES['image']:
        file = request.FILES['image']
        task = process_download_with_progress.delay(file, 10000)
        print(task.task_id)
        return render(request, 'index.html', {'task_id': 'task.task_id'})
    return render(request, 'upload.html')

# def simple_upload(request):
#     if request.method == 'POST' and request.FILES['image']:
#         myfile = request.FILES['image']
#         fs = FileSystemStorage()
#         buffer = io.BytesIO()
#         for chunk in read_chunk(myfile.file, 125):
#             buffer.write(chunk)
#         buffer.seek(0)
#         buffer = io.BytesIO(buffer.getvalue())
#         #buffer = buffer.getvalue()
#         image = ImageFile(buffer, 'name.jpg')
#         fs.save(myfile.name, image)
#         return render(request, 'index.html')
#     return render(request, 'upload.html')
        

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
def signout(request):
    logout(request)
    return redirect('index')


@login_required()
def profile(request):
    if request.method == 'POST':
        # current_user = UserProfile.objects.get(username=request.user)
        form = UserDetailsUpdate(request.POST, instance=request.user.profile)
        if form.is_valid():
            form.save(commit=True)
            return redirect('profile')
    form = UserDetailsUpdate(instance=request.user.profile)
    return render(request, 'profile.html', {'form': form})



def registration(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            password = form.cleaned_data.get('password')
            user.set_password(password)
            user.save()
            return redirect('signin')
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




        #file_size = file.size
        # fs = FileSystemStorage()
        # buffer = io.BytesIO()    
        # chunk_size = 0
        # for chunk in file.chunks():
        #     chunk_size += sys.getsizeof(chunk)
        #     buffer.write(chunk)
        # print(f'chunks sizes is {chunk_size} and file size is {file_size}')
        # buffer.seek(0)
        # image = ImageFile(buffer, name=file.name)
        # fs.save(file.name, content=image)
        #print(image.size)
        #task = process_download.delay(image)
        #print(task.task_id)