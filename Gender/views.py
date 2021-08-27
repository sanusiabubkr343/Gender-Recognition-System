from django.shortcuts import render
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
from django.core.files.storage import FileSystemStorage
from . import settings
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import os


# python .\manage.py runserver


def home(request):
    return render(request, "home.html")


def result(request):
    model = load_model(".\gender_detection.h5")

    myfile = request.FILES["myfile"]
    fs = FileSystemStorage(location=(settings.MEDIA_ROOT))
    filename = fs.save(myfile.name, myfile)
    file_url = fs.url(filename)
    img_url = os.path.join(settings.MEDIA_ROOT, filename)
    image_ = plt.imread(img_url)
    image_ = cv2.resize(image_, (128, 128))/255
    image_ = np.expand_dims(image_, axis=0)
    p_ = model.predict(image_)
    print(p_)
    class_ = "Male" if p_ < 0.4 else "Female"
    proba_= p_[0][0]*100 if p_> 0.5 else (1-p_[0][0])*100
    return render(request, "result.html", context={"img": file_url, "class": class_, 'proba': round(proba_, 2)})
