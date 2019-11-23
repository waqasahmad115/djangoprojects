
from django import forms
from django.forms import ModelForm
from .models import MyPoiRecord


class MyPoiRecordForm(ModelForm):
    class Meta:
        model = MyPoiRecord
        exclude = ()
        









