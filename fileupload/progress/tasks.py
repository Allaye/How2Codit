import time
from celery import shared_task
from celery_progress.backend import ProgressRecorder



@shared_task(bind=True)
def process_download(self):
    progress_recoder = ProgressRecorder(self)
    print('Start')
    for i in range(5):
        time.sleep(1)
        print(i + 1)
        progress_recoder.set_progress(i + 1, 5, description='downloading')
    print('End')
    return 'done sleeping'