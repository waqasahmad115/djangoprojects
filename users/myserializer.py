from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Profile,SecurityPersonnel
from POI_Record.models import MyPoiRecord
from camera.models import MyDetected_Poi
class ProfileSerializer(serializers.ModelSerializer):
        class Meta:
            model=Profile
            fields="__all__"

class SecurityPersonnelSerializer(serializers.ModelSerializer):
        class Meta:
            model=SecurityPersonnel
            fields="__all__"
class MyPoiRecordSerializer(serializers.ModelSerializer):
        class Meta:
            model=MyPoiRecord
            fields="__all__"
class MyDetected_PoiSerializer(serializers.ModelSerializer):
        class Meta:
            model=MyDetected_Poi
            fields="__all__"
            
class UserSerializer(serializers.ModelSerializer):
        class Meta:
            model=User
            fields= ['username','first_name','last_name','email', 'password',]