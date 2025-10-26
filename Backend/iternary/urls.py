from django.urls import path
from . import views

urlpatterns = [
    path('upload-video/', views.handle_mp4_upload, name='upload_video'),
]
