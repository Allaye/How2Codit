from django.shortcuts import render
from django.http import HttpResponse
from .tasks import send_email

# Create your views here.
def index(request):
    send_email.delay()
    return HttpResponse('<h1>Task 1 of sending email with celery!</h1>')
