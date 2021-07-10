from time import sleep
from django.shortcuts import render
from progress.forms import UserForm
# Create your views here.
def index(request):
    form = UserForm()

    context = {'form':form}
    return render(request, 'index.html', context)