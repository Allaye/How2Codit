from time import sleep
from celery import shared_task
from django.core.mail import send_mail


@shared_task()
def sleepy(duration):
    sleep(duration)
    return None


@shared_task()
def send_email():
    sleep(10)
    send_mail('Celery email task','This is proof that the email word', 'ekonlinehub@gmail.com', ['ekonlinehub@gmail.com'])
    return None