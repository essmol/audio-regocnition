from django.urls import path
from .views import predict_image

app_name='doctor'

urlpatterns = [
    path('predict/', predict_image, name='predict_image'),
]