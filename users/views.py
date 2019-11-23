from django.core.mail import send_mail
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse
from .forms import UserRegisterForm,ControlRegisterForm
from django.contrib import auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .models import User, SecurityPersonnel,Profile,ControlRoomOperator
from POI_Record.models import MyPoiRecord
from camera .models import MyDetected_Poi
from .myserializer import ProfileSerializer ,SecurityPersonnelSerializer,MyPoiRecordSerializer,UserSerializer,MyDetected_PoiSerializer
from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm
from django.core.mail import send_mail
from django.conf import settings
from rest_framework import viewsets
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
from PIL import Image
from numpy import asarray, expand_dims
import imutils
import pickle
import time
import cv2
from PIL import Image
from keras.models import load_model
from keras import backend as K
K.clear_session()
#detected_poi= {}
def base(request):
    # print("[INFO] loading face detector...")
    # protoPath = "face_detection/deploy.prototxt"
    # modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    # detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # # load our serialized face embedding model from disk
    # print("[INFO] loading face recognizer...")
    # embedder = load_model('facenet/facenet_keras.h5')
    # # load the actual face recognition model along with the label encoder
    # rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
    # recognizer = pickle.loads(open(rec_path, "rb").read())
    # le = pickle.loads(open(le_path, "rb").read())
    # conf_threshold = 0.5
    # required_size = (160, 160)
    # # initialize the video stream, then allow the camera sensor to warm up
    # print("[INFO] starting video stream...")
    # cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video you guys have path of vedio 
    # time.sleep(2.0)
    # # start the FPS throughput estimator
    # fps = FPS().start()
    # # loop over frames from the video file stream
    # while(cap.isOpened()):
    #     _, frame = cap.read()
    #     frame = imutils.resize(frame, width=600)
    #     (h, w) = frame.shape[:2]
    #     # construct a blob from the image
    #     imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    #         (104.0, 177.0, 123.0), swapRB=False, crop=False)
    #     # apply OpenCV's deep learning-based face detector to localize
    #     # faces in the input image
    #     detector.setInput(imageBlob)
    #     detections = detector.forward()
    #     # loop over the detections
    #     for i in range(0, detections.shape[2]):
    #         # extract the confidence (i.e., probability) associated with  the prediction
    #         confidence = detections[0, 0, i, 2]
    #         # filter out weak detections
    #         if confidence > conf_threshold:
    #             # compute the (x, y)-coordinates of the bounding box for the face
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")
    #             # extract the face ROI
    #             face = frame[startY:endY, startX:endX]
    #             (fH, fW) = face.shape[:2]

    #             # ensure the face width and height are sufficiently large
    #             if fW < 20 or fH < 20:
    #                 continue
    #             image = Image.fromarray(face) 
    #             # resize pixels to the model size
    #             image = image.resize(required_size)
    #             face_array = asarray(image)
    #             face_pixels = face_array.astype('float32')
    #             # standardize pixel values across channels (global)
    #             mean, std = face_pixels.mean(), face_pixels.std()
    #             face_pixels = (face_pixels - mean) / std
    #             # transform face into one sample
    #             samples = expand_dims(face_pixels, axis=0)
    #             # make prediction to get embedding
    #             yhat = embedder.predict(samples)
    #             vec = yhat[0]
    #             vec = np.reshape(vec, (1, 128))
    #             # perform classification to recognize the face
    #             preds = recognizer.predict_proba(vec)[0]
    #             j = np.argmax(preds)
    #             proba = preds[j]
    #             name = le.classes_[j]
    #             # draw the bounding box of the face along with the associated probability
    #             text = "{}: {:.2f}%".format(name, proba * 100)
    #             y = startY - 10 if startY - 10 > 10 else startY + 10
    #             cv2.rectangle(frame, (startX, startY), (endX, endY),
    #                 (0, 0, 255), 2)
    #             cv2.putText(frame, text, (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #             # update the FPS counter

    #             if (name.lower() == "unknown"):
    #                 #print("hello")
                
    #                 continue
    #             else:
    #                 time_str = time.asctime()  
    #                 x = time_str.split()
    #                 detect_date=str(x[1])+" "+str(x[2])+" "+str(x[-1]) 
    #                 detect_time = x[3:-1][0]
                    
    #                 if (name in detected_poi) and (detected_poi[name][0] == detect_date) and (detected_poi[name][1][0:2] == detect_time[0:2]) and (detected_poi[name][1][3:5] == detect_time[3:5]) and (int(detected_poi[name][1][6:8]) == int(detect_time[6:8])-5):
    #                     print("inside")
    #                     continue
    #                 else:
    #                     if name in detected_poi:
    #                         detected_poi[name][0] = detect_date
    #                         detected_poi[name][1] = detect_time
    #                         detected_poi[name][2] = detected_poi[name][2] + 1
    #                     else:
    #                         detected_poi.update({name:[detect_date,detect_time,1]})
    #                     img=Image.fromarray(frame)
    #                     img = img.save(name + str(detected_poi[name][2])+".png")
            
    #             fps.update()
    #             # show the output frame
    #             img=Image.fromarray(frame,"RGB")
    #             cv2.imshow("Frame", frame)
    #             key = cv2.waitKey(1) & 0xFF
    #             # if the `q` key was pressed, break from the loop
    #             if cv2.waitKey(1) & 0xFF == ord('q'):break
    # # stop the timer and display FPS information
    # fps.stop()
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # cap.release()
    # cv2.destroyAllWindows()                                                                                         
    return render(request,'base.html')

def vedio(request):
    print("[INFO] loading face detector...")
    protoPath = "face_detection/deploy.prototxt"
    modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = load_model('facenet/facenet_keras.h5')
    # load the actual face recognition model along with the label encoder
    rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
    recognizer = pickle.loads(open(rec_path, "rb").read())
    le = pickle.loads(open(le_path, "rb").read())
    conf_threshold = 0.5
    required_size = (160, 160)
    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video you guys have path of vedio 
    time.sleep(2.0)
    # start the FPS throughput estimator
    fps = FPS().start()
    # loop over frames from the video file stream
    while(cap.isOpened()):
        _, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with  the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > conf_threshold:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue
                image = Image.fromarray(face) 
                # resize pixels to the model size
                image = image.resize(required_size)
                face_array = asarray(image)
                face_pixels = face_array.astype('float32')
                # standardize pixel values across channels (global)
                mean, std = face_pixels.mean(), face_pixels.std()
                face_pixels = (face_pixels - mean) / std
                # transform face into one sample
                samples = expand_dims(face_pixels, axis=0)
                # make prediction to get embedding
                yhat = embedder.predict(samples)
                vec = yhat[0]
                vec = np.reshape(vec, (1, 128))
                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                # draw the bounding box of the face along with the associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                # update the FPS counter
                fps.update()
                # show the output frame
                img=Image.fromarray(frame,"RGB")
                img.save("my.png")
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key was pressed, break from the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):break
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()                                                                                                          	       
    return render(request,'vedio.html')


# def dashboard(request):
#     return render(request,'dashboard.html')

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            User.is_staff=True
            save_it=form.save()
            username = form.cleaned_data.get('username')
            form_email=form.cleaned_data.get('email')
            subject = 'Thank you for registering to our site Automated Survaillance System'
            message = ' Your Account Will be verify within maximum of the 1 Day duration.An Email will be sent to your account ,by clicking on activation link .you will be able to login to site .'
            email_from = settings.EMAIL_HOST_USER
            to_list = [email_from,form_email]
            send_mail( subject, message, email_from, to_list,fail_silently=False)
        #   messages.success(request, f'Account created for {username}!')
            messages.success(request, f'Email has been sent to your Account .Please Check it First !')
            return redirect('base')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})

@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST,
                            
                                   request.FILES,
                                   instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, f'Your account has been updated!')
            return redirect('profile')

    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form': u_form,
        'p_form': p_form
    }

    return render(request, 'users/profile.html', context)
    
    
    
def signup(request):
    if request.method == "POST":
        # to create a user
        if request.POST['pass'] == request.POST['passwordagain']:
            # both the passwords matched
            # now check if a previous user exists
            try:
                user = User.objects.get(username=request.POST['uname'])
                return render(request, 'users/signup.html', {'error': "Username Has Already Been Taken"})
            except User.DoesNotExist:
                user = User.objects.create_user(username= request.POST['uname'],password= request.POST['pass'])
                phnum = request.POST['phone']
                zone = request.POST['zone']
                stime=request.POST['stime']
                etime=request.POST['etime']

                Personel=  SecurityPersonnel(phone_number=phnum, zone_area=zone, user=user,start_time=stime, end_time=etime)
                Personel.save()
                auth.login(request, user)
                
             #  return HttpResponse("Signned Up !")
                messages.success(request, f'Email has been sent to your Account !')
                return redirect('base')
        else:
            return render(request, 'users/signup', {'error': "Passwords Don't Match"})
    else:
        return render(request, 'users/signup.html')


def user_list(request):
    return render(request, 'users/user_list.html')

class ProfileViewSet(viewsets.ModelViewSet):
        queryset=Profile.objects.all().order_by('-id')
        serializer_class=ProfileSerializer
class SecurityPersonnelViewSet(viewsets.ModelViewSet):
        queryset=SecurityPersonnel.objects.all().order_by('-id')
        serializer_class=SecurityPersonnelSerializer
class MyPoiRecordViewSet(viewsets.ModelViewSet):
        queryset=MyPoiRecord.objects.all().order_by('-id')
        serializer_class=MyPoiRecordSerializer
class MyDetected_PoiViewSet(viewsets.ModelViewSet):
        queryset=MyDetected_Poi.objects.all().order_by('-id')
        serializer_class=MyDetected_PoiSerializer
class UserViewSet(viewsets.ModelViewSet):
        queryset=User.objects.all().order_by('-id')
        serializer_class=UserSerializer




# from imutils.video import VideoStream
# from imutils.video import FPS
# import numpy as np
# from numpy import asarray, expand_dims
# import imutils
# import pickle
# import time
# import cv2
# from PIL import Image
# from keras.models import load_model

# def base(request):
#     protoPath = "face_detection/deploy.prototxt"
#     modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
#     detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#     # load our serialized face embedding model from disk
# #   print("[INFO] loading face recognizer...")
#     embedder = load_model('facenet/facenet_keras.h5')
#     # load the actual face recognition model along with the label encoder
#     rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
#     recognizer = pickle.loads(open(rec_path, "rb").read())
#     le = pickle.loads(open(le_path, "rb").read())
#     conf_threshold = 0.5
#     required_size = (160, 160)
#     # initialize the video stream, then allow the camera sensor to warm up
# #   print("[INFO] starting video stream...")
#     cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video you guys have
#     time.sleep(2.0)
#     # start the FPS throughput estimator
#     fps = FPS().start()
#     # loop over frames from the video file stream
#     while(cap.isOpened()):
#         _, frame = cap.read()
#         frame = imutils.resize(frame, width=600)
#         (h, w) = frame.shape[:2]
#         # construct a blob from the image
#         imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
#             (104.0, 177.0, 123.0), swapRB=False, crop=False)
#         # apply OpenCV's deep learning-based face detector to localize
#         # faces in the input image
#         detector.setInput(imageBlob)
#         detections = detector.forward()
#         # loop over the detections
#         for i in range(0, detections.shape[2]):
#             # extract the confidence (i.e., probability) associated with  the prediction
#             confidence = detections[0, 0, i, 2]
#             # filter out weak detections
#             if confidence > conf_threshold:
#                 # compute the (x, y)-coordinates of the bounding box for the face
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 # extract the face ROI
#                 face = frame[startY:endY, startX:endX]
#                 (fH, fW) = face.shape[:2]

#                 # ensure the face width and height are sufficiently large
#                 if fW < 20 or fH < 20:
#                     continue
#                 image = Image.fromarray(face) 
#                 # resize pixels to the model size
#                 image = image.resize(required_size)
#                 face_array = asarray(image)
#                 face_pixels = face_array.astype('float32')
#                 # standardize pixel values across channels (global)
#                 mean, std = face_pixels.mean(), face_pixels.std()
#                 face_pixels = (face_pixels - mean) / std
#                 # transform face into one sample
#                 samples = expand_dims(face_pixels, axis=0)
#                 # make prediction to get embedding
#                 yhat = embedder.predict(samples)
#                 vec = yhat[0]
#                 vec = np.reshape(vec, (1, 128))
#                 # perform classification to recognize the face
#                 preds = recognizer.predict_proba(vec)[0]
#                 j = np.argmax(preds)
#                 proba = preds[j]
#                 name = le.classes_[j]
#                 # draw the bounding box of the face along with the associated probability
#                 text = "{}: {:.2f}%".format(name, proba * 100)
#                 y = startY - 10 if startY - 10 > 10 else startY + 10
#                 cv2.rectangle(frame, (startX, startY), (endX, endY),
#                     (0, 0, 255), 2)
#                 cv2.putText(frame, text, (startX, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#                 # update the FPS counter
#                 fps.update()
#                 # show the output frame
#                 cv2.imshow("Frame", frame)
#                 key = cv2.waitKey(1) & 0xFF
#                 # if the `q` key was pressed, break from the loop
#                 if cv2.waitKey(1) & 0xFF == ord('q'):break
#     # stop the timer and display FPS information
#     fps.stop()
# #   print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# #   print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#     cap.release()
#     cv2.destroyAllWindows()  
#     return render(request,'view_live_stream.html')
