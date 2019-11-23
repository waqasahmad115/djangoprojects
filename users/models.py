from django.db import models
#from django.contrib.auth.models import AbstractUser 
from django.contrib.auth.models import User
from PIL import Image
from django.dispatch import receiver
# Create your models here.
# class User(AbstractUser):
#     is_security_personnel=models.BooleanField(default=False)
#     is_control_room_operator=models.BooleanField(default=False)

#     def __str__(self):
#         return self.username
        
class SecurityPersonnel(models.Model):
    user=models.OneToOneField(User,on_delete=models.CASCADE)
    phone_number=models.CharField(max_length=15)
    zone_area=models.CharField(max_length=30)
    start_time=models.CharField(max_length=30)
    end_time=models.CharField(max_length=30 )

class ControlRoomOperator(models.Model):
    user=models.OneToOneField(User,on_delete=models.CASCADE)
    operator_area=models.CharField(max_length=30)
    start_time=models.CharField(max_length=30)
    end_time=models.CharField(max_length=30)
    def __str__(self):
        return self.user.username
# @receiver(post_save, sender=User)
# def create_user_profile(sender, instance, created, **kwargs):
# 	print('****', created)
# 	if instance.is_security_personnel:
# 		SecurityPersonnel.objects.get_or_create(user = instance)
# 	else:
# 		ControlRoomOperator.objects.get_or_create(user = instance)
	
# @receiver(post_save, sender=User)
# def save_user_profile(sender, instance, **kwargs):
# 	print('_-----')	
# 	# print(instance.internprofile.bio, instance.internprofile.location)
# 	if instance.is_security_personnel:
# 		instance.Security_Personnel.save()
# 	else:
# 		ControlRoomOperator.objects.get_or_create(user = instance)
	
        
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')

    def __str__(self):
        return f'{self.user.username} Profile'


    def save(self, *args, **kwargs):
      super(Profile, self).save(*args, **kwargs)

      img = Image.open(self.image.path)

      if img.height > 300 or img.width > 300:
            output_size = (300, 300)
            img.thumbnail(output_size)
            img.save(self.image.path)
