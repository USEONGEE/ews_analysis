from django.db import models

# Create your models here.
from django.db import models
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
# Create your models here.

class FileUpload(models.Model):
  title = models.TextField(max_length=40, null=True)
  imgfile = models.ImageField(null=True, upload_to="", blank=True)
  content = models.TextField()