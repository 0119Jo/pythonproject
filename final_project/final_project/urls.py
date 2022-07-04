"""final_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls.conf import include
from myapp import views
from django.conf import settings
from django.conf.urls.static import static
# pip3 install pillow
urlpatterns = [
    path('admin/', admin.site.urls),
    path('portfolio', views.lda_model), 
    path('', views.Main),
    #path('about', views.yoloModel), 
    path('upload', views.Upload),
    path('chart_word', views.chart_word), 
    path('/myapp/', include('myapp.urls')),
    
]
urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
