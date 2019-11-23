from django.contrib import admin

# Register your models here.
from django.contrib import admin
from.models import MyCamera ,MyZone,MyDetected_Poi,MyCity
from django.contrib.admin.options import ModelAdmin 
# Register your models here.
class MyCityAdmin(ModelAdmin):
    list_display=["CID","city_name"]
    search_fields=["city_name"]
    list_filter=["city_name"]
admin.site.register(MyCity,MyCityAdmin)
class MyCameraAdmin(ModelAdmin):
    list_display=["CCID","Logitude","Latitude"]
    search_fields=["Logitude","Latitude"]
    list_filter=["Logitude","Latitude"]
admin.site.register(MyCamera,MyCameraAdmin)

class MyZoneAdmin(ModelAdmin):
    list_display=["ZID","Zone_area_name"]
    search_fields=["Zone_area_name"]
    list_filter=["Zone_area_name"]
admin.site.register(MyZone,MyZoneAdmin)
class MyDetected_PoiAdmin(ModelAdmin):
    list_display=["DID","detected_image","date_time"]
    list_filter=["date_time"]
admin.site.register(MyDetected_Poi,MyDetected_PoiAdmin)
