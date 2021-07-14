import time
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse


# @shared_task(bind=True)
# def process_download(self):
#     progress_recoder = ProgressRecorder(self)
#     print('Start')
#     for i in range(5):
#         time.sleep(1)
#         print(i + 1)
#         progress_recoder.set_progress(i + 1, 5, description='downloading')
#     print('End')
#     return 'done sleeping'

@shared_task(bind=True)
def process_download(self, image_file):
    print('Upload: Task Started')
    time.sleep(50)
    # fs = FileSystemStorage()
    # filename = fs.save(image_file.name, image_file)
    # uploaded_file_url = fs.url(filename)
    return 'Done'

    # return  uploaded_file_url

# def update_progress(self, proc):

#     process_recorder  = ProgressRecorder(self)
    






# def simple_upload(request):
#     if request.method == 'POST' and request.FILES['image']:
#         image = request.FILES['image']
#         fs = FileSystemStorage()
#         filename = fs.save(image.name, image)
#         uploaded_file_url = fs.url(filename)
#         return uploaded_file_url