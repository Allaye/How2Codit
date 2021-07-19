import io
import sys
import time
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.core.files.images import ImageFile


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
    process_recoder = ProgressRecorder(self)
    print('Upload: Task Started')
    # time.sleep(50)
    fs = FileSystemStorage()
    buffer = io.BytesIO()
    image_size = sys.getsizeof(image_file)
    chunk_size = 0
    for chunk in image_file.chunks():
        chunk_size += sys.getsizeof(chunk)
        buffer.write(chunk)
        process_recoder.set_progress(chunk_size, image_size, description=f'uploaded {chunk_size} bytes of the file')
    buffer.seek(0)
    image = ImageFile(buffer, name=image_file.name)
    fs.save(image_file.name, image)
    return 'Done'

# file = request.FILES['image']
#         file_size = file.size
#         fs = FileSystemStorage()
#         buffer = io.BytesIO()
#         chunk_size = 0
#         for chunk in file.chunks():
#             chunk_size += sys.getsizeof(chunk)
#             buffer.write(chunk)
#         print(f'chunks sizes is {chunk_size} and file size is {file_size}')
#         buffer.seek(0)
#         image = ImageFile(buffer, name=file.name)
#         fs.save(file.name, content=image)





# def simple_upload(request):
#     if request.method == 'POST' and request.FILES['image']:
#         image = request.FILES['image']
#         fs = FileSystemStorage()
#         filename = fs.save(image.name, image)
#         uploaded_file_url = fs.url(filename)
#         return uploaded_file_url


# @shared_task(bind=True)
# def ccount(self):
#     progess = ProgressRecorder(self)
#     for i in range(30):
#         time.sleep(i)
#         progess.set_progress(i+1, 10, description='testing')
#     return 'Job done'
