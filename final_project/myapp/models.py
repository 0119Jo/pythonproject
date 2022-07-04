from django.db import models

class Profile(models.Model):
    image = models.ImageField(upload_to='images/')
    
    def __str__(self):
        return self.title