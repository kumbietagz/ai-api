from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import joblib
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import wget

from django.views.decorators.csrf import csrf_exempt


forest_job = joblib.load('ML/RandomForest.pkl')
# Create your views here.

@csrf_exempt
@api_view(['POST'])
def predict(request):
    data = request.data
    print(data["nitrogen"])
    crop = forest_job.predict([[data["nitrogen"], data["phosphorous"], data["potassium"], data["temperature"], data["humidity"], data["ph"], data["rainfall"]]])
    return Response({"crop":crop[0]})

default_image_size = tuple((64, 64))

def convert_image_to_array(image_dir):
    try:

        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print("Error :", e)
        return None

@csrf_exempt
@api_view(['POST'])
def classify(request):
    data = request.data
    print(data["image"])
    filename = wget.download(data["image"])
    img = convert_image_to_array(filename)
    img = np.array(img, dtype=np.float16) / 225.0
    img.resize(1,64,64,3)
    model = load_model('ML/trained_model.h5')
    res = model.predict(img)
    value = np.argmax(res)
    if (value == 0):
        dictionary = {'plant' : 'apple', 'disease': 'scab'}
    elif (value == 1):
        dictionary = {'plant' : 'apple', 'disease': 'black rot'}        
    elif (value == 2):
        dictionary = {'plant' : 'apple', 'disease': 'cedar apple rust'}
    elif (value == 3):
        dictionary = {'plant' : 'apple', 'disease': 'healthy'}
    elif (value == 4):
        dictionary = {'plant' : 'blueberry', 'disease': 'healthy'}
    elif (value == 5):
        dictionary = {'plant' : 'cherry', 'disease': 'healthy'}
    elif (value == 6):
        dictionary = {'plant' : 'cherry', 'disease': 'powdery mildew'}
    elif (value == 7):
        dictionary = {'plant' : 'corn', 'disease': 'cercospora leaf spot gray leaf spot'}
    elif (value == 8):
        dictionary = {'plant' : 'corn', 'disease': 'common rust'}
    elif (value == 9):
        dictionary = {'plant' : 'corn', 'disease': 'healthy'}
    elif (value == 10):
        dictionary = {'plant' : 'corn', 'disease': 'nothern leaf blight'}        
    elif (value == 11):
        dictionary = {'plant' : 'grape', 'disease': 'black rot'}
    elif (value == 12):
        dictionary = {'plant' : 'grape', 'disease': 'esca (black measles)'}
    elif (value == 13):
        dictionary = {'plant' : 'grape', 'disease': 'healthy'}
    elif (value == 14):
        dictionary = {'plant' : 'grape', 'disease': 'leaf blight (isariopsis_leaf_spot)'}
    elif (value == 15):
        dictionary = {'plant' : 'orange', 'disease': 'orange haunglongbing (citrus greening)'}
    elif (value == 16):
        dictionary = {'plant' : 'peach', 'disease': 'bacterial spot'}
    elif (value == 17):
        dictionary = {'plant' : 'peach', 'disease': 'healthy'}
    elif (value == 18):
        dictionary = {'plant' : 'pepper, bell', 'disease': 'bacterial spot'}
    elif (value == 19):
        dictionary = {'plant' : 'pepper, bell', 'disease': 'healthy'}
    elif (value == 20):
        dictionary = {'plant' : 'potato', 'disease': 'early blight'}
    elif (value == 21):
        dictionary = {'plant' : 'potato', 'disease': 'healthy'}
    elif (value == 22):
        dictionary = {'plant' : 'potato', 'disease': 'late blight'}
    elif (value == 23):
        dictionary = {'plant' : 'raspberry', 'disease': 'healthy'}
    elif (value == 24):
        dictionary = {'plant' : 'soybean', 'disease': 'healthy'}
    elif (value == 25):
        dictionary = {'plant' : 'squash', 'disease': 'powdery mildew'}
    elif (value == 26):
        dictionary = {'plant' : 'strawberry', 'disease': 'healthy'}
    elif (value == 27):
        dictionary = {'plant' : 'strawberry', 'disease': 'leaf scorch'}
    elif (value == 28):
        dictionary = {'plant' : 'tomato', 'disease': 'bacterial spot'}
    elif (value == 29):
        dictionary = {'plant' : 'tomato', 'disease': 'early blight'}
    elif (value == 30):
        dictionary = {'plant' : 'tomato', 'disease': 'healthy'}
    elif (value == 31):
        dictionary = {'plant' : 'tomato', 'disease': 'late blight'}
    elif (value == 32):
        dictionary = {'plant' : 'tomato', 'disease': 'leaf mold'}
    elif (value == 33):
        dictionary = {'plant' : 'tomato', 'disease': 'septoria leaf spot'}
    elif (value == 34):
        dictionary = {'plant' : 'tomato', 'disease': 'spider mites two-spoted spider mite'}
    elif (value == 35):
        dictionary = {'plant' : 'tomato', 'disease': 'target spot'}
    elif (value == 36):
        dictionary = {'plant' : 'tomato', 'disease': 'tomato mosaic virus'}
    elif (value == 37):
        dictionary = {'plant' : 'tomato', 'disease': 'tomato yellow leaf curl virus'}
    
    images = request.FILES.getlist('images')
    print(images)


    print(np.argmax(res))
    return Response(dictionary)
    
    



    