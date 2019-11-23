from django.db import models
import os
# Create your models here.
from django.utils import timezone
path = "static/POI"
choice = (
    ('Choose threat level', 'Choose threat level'),
    ('Low', 'Low'),
    ('Medium', 'Medium'),
    ('High', 'High'),
    ('Very High', 'Very High')
)
# Create your models here.
class MyPoiRecord(models.Model):
    name = models.CharField(max_length=30,null=True)
    age = models.IntegerField(null=True)
    DOB= models.DateField()
    comments = models.CharField(max_length=255)
    threat_level=models.CharField(max_length=50, choices=choice, default='None')
    image1=models.ImageField(upload_to='POI/ali')
    image2=models.ImageField(upload_to='POI/ali')
    image3=models.ImageField(upload_to='POI/ali')
    image4=models.ImageField(upload_to='POI/ali')
    image5=models.ImageField(upload_to='POI/ali')
    image6=models.ImageField(upload_to='POI/ali')
    image7=models.ImageField(upload_to='POI/ali')
    image8=models.ImageField(upload_to='POI/ali')
    image9=models.ImageField(upload_to='POI/ali')
    image10=models.ImageField(upload_to='POI/ali')
    def __str__(self):
        return "{0}".format(self.name)

    class Meta:
        db_table = "poirecord_mypoirecord"  
       
