from django.db import models

# Create your models here.
from django.db import models
from POI_Record.models import MyPoiRecord
# Create your models here.
from django.utils import timezone

class MyCity(models.Model):
    CID=models.IntegerField(unique=True)
    city_name=models.CharField(max_length=255)
    
    class Meta:
        db_table = "camera_mycity"  

class MyZone(models.Model):
    ZID=models.IntegerField(unique=True)
    CID= models.ForeignKey(MyCity,on_delete=models.CASCADE)
    Zone_area_name=models.CharField(max_length=255)
    radius= models.DecimalField(decimal_places=2, max_digits=12)
    Latitude= models.DecimalField(decimal_places=2, max_digits=12)
    Logitude= models.DecimalField(decimal_places=2, max_digits=12)
    
   
   
    
    class Meta:
        db_table = "camera_myzone"  
    

class MyCamera(models.Model):
    CCID=models.IntegerField(unique=True)
    Logitude= models.DecimalField(decimal_places=2, max_digits=12)
    Latitude= models.DecimalField(decimal_places=2, max_digits=12)
    ZoneID= models.ForeignKey(MyZone,on_delete=models.CASCADE)
    def __str__(self):
        return "{0} {1}".format(self.Logitude, self.Latitude)

    class Meta:
        db_table = "camera_mycamera"  
   
   
class MyDetected_Poi(models.Model):
    DID=models.IntegerField(unique=True)
    cameraID= models.ForeignKey(MyCamera,on_delete=models.CASCADE)
    poiID= models.ForeignKey(MyPoiRecord,on_delete=models.CASCADE)
    detected_image=models.ImageField()
    date_time = models.DateField()
    class Meta:
        db_table = "camera_mydetected_poi"  
   






