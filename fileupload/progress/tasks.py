import time
from celery import shared_task
from celery_progress.backend import ProgressRecorder



@shared_task(bind=True)
def go_to_sleep(self, duration):
    sleep(duration)
    return 'done sleeping'