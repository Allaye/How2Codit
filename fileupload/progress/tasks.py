import io
import math
import sys
import time
from PIL import Image
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
    fs = FileSystemStorage()
    buffer = io.BytesIO()
    chunk_size = 0
    for chunk in image_file.chunks():  
        chunk_size += len(chunk)      
        buffer.write(chunk)
        process_recoder.set_progress(chunk_size, image_file.size, description=f'uploaded {chunk_size} bytes of the file')
    buffer.seek(0)
    image = ImageFile(buffer, name=image_file.name)
    fs.save(image_file.name, content=image)
    return 'Done'

@shared_task(bind=True)
def process_download_with_progress(self, image_file, length):
    process_recoder = ProgressRecorder(self)
    print('Upload: Task Started')
    fs = FileSystemStorage()
    buffer = io.BytesIO()
    chunk_size = 0
    for chunk in read_chunk(image_file.file, length):  
        chunk_size += 1      
        buffer.write(chunk)
        if chunk_size == 1:
            length = math.ceil((len(image_file.file.getvalue()))/10000)
        process_recoder.set_progress(chunk_size, length, description=f'uploaded {chunk_size*length} bytes of the file')
    buffer.seek(0)
    image = io.BytesIO(buffer.getvalue())
    image_file = ImageFile(image, name=image_file.name)
    fs.save(image_file.name, image)
    return 'Done'
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
def read_chunk(file_object, chunk_size=125):
    while True:
        file =  file_object.read(chunk_size)
        if not file:
            break
        yield file



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
