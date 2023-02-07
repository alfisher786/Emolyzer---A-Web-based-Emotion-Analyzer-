from django.contrib import admin
from django.urls import path
from NLP import views

urlpatterns = [
    path('', views.index, name='index'),
    path('started/', views.started, name='started'),
    path('text/', views.text, name='text'),
    path('twitter/', views.twitter, name='twitter'),
    path('amazon/', views.amazon, name='amazon'),
    path('contact/', views.contact, name='contact'),
    path('result/', views.result, name='result'),
    path('result2/', views.result2, name='result2'),
    path('result3/', views.result3, name='result3')
    ]
