from django.shortcuts import render,redirect
#from . import extract_embeddings
from time import sleep
from POI_Record.models import MyPoiRecord
from django.contrib import messages
# Create your views here.
from django.shortcuts import render

from django.http import HttpResponse
from django.shortcuts import render
from POI_Record.forms import MyPoiRecordForm
def view_poi(request):
    poi=MyPoiRecord.objects.all()
    poi=MyPoiRecord.objects.order_by('id')
    return render(request,'POI_Record/view_poi.html',{'poi':poi})
# def change(request):
#     extract_embeddings.main()
#     return render(request, "POI_Record/change.html")

# #change

def addpoi_form(request):
    name= request.POST['name']
    print("hello world")
    age= request.POST['age']
    DOB=request.POST['DOB'] 
    comments=request.POST['comments']
    threat_level=request.POST['threat_level']
    image1=request.POST['image1']
    image2=request.POST['image2']
    image3=request.POST['image3']
    image4=request.POST['image4']
    image5=request.POST['image5']
    image6=request.POST['image6']
    image7=request.POST['image7']
    image8=request.POST['image8']
    image9=request.POST['image9']
    image10=request.POST['image10']
    p=MyPoiRecord(name=name,age=age,DOB=DOB,comments=comments,threat_level=threat_level,image1=image1,image2=image2,image3=image3,image4=image4,image5=image5,image6=image6,image7=image7,image8=image8,image9=image9,image10=image10)
    p.save()
    messages.success(request, f'POI Record has been Added Successfully  !')
    return render(request,'POI_Record/addpoi.html')



def addpoi(request):
    return render(request, 'POI_Record/addpoi.html')
# from imutils import paths
# import numpy as np
# import imutils
# import pickle
# import cv2
# import os
# from threading import Thread
# from keras.models import load_model
# from numpy import asarray, expand_dims
# from PIL import Image
# name = "akash"
# def change(request):
#     if request.method == 'POST':
#         form=MyPoiRecordForm(request.POST)
    
#         name= request.POST.get('name ')
#         age= request.POST.get('age')
#         dob=request.POST.get('dob')
#         comments=request.POST.get('comments')    
#         threat=request.POST.get('threat')
#         image1=request.POST.get('iamge1')
#         image2=request.POST.get('iamge2')
#         image3=request.POST.get('iamge3')
#         image4=request.POST.get('iamge4')
#         image5=request.POST.get('iamge5')
#         image6=request.POST.get('iamge6')
#         image7=request.POST.get('iamge7')
#         image8=request.POST.get('iamge8')
#         image9=request.POST.get('iamge9')
#         image10=request.POST.get('iamge10')
#         p=MyPoiRecord(name=name,age=age,comments=comments,threat=threat,iamge1=iamge1,iamge2=iamge2,iamge3=iamge3,iamge4=iamge4,iamge5=iamge5,iamge6=iamge6,iamge7=iamge7,iamge8=iamge8,iamge9=iamge9,image10=image10)
#         p.save()
#         messages.success(request, f'Email has been sent to your Account .Please Check it First !')
#         return redirect('base')
#     else:
#         form=MyPoiRecordForm()

#      return render(request, "POI_Record/change.html")




