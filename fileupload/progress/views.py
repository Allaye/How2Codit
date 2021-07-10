from time import sleep
from django.shortcuts import render
from progress.forms import UserForm
from progress.models import User



# Create your views here.
def index(request):
    

    form = UserForm()
    return render(request, 'index.html', {'form': form})