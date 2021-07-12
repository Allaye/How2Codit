from django.db import models
from django.contrib.auth.models import User




# Create your models here.
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    sex = models.CharField(max_length=20, blank=True)
    website = models.URLField(blank=True)
    image = models.ImageField(blank=True)

    def __str__(self):
        return self.user.username

    def save(self):
        super().save()