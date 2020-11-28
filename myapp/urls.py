from django.urls import path
from . import views

urlpatterns = [
    path('calculate/', views.calculate),
    path('getOpenid/', views.getopenid),
    path('login/', views.login),
    path('modify/', views.modify),
    path('AI_report/', views.AI_report),
    path('AI_image/', views.AI_image),
    path('AI_scale1/', views.AI_scale1),
    path('AI_scale2/', views.AI_scale2),
    path('report_content/', views.report_content),
    path('mutifile/', views.mutifile),
    path('myMocaHistory/', views.myMocaHistory),
    path('patientMyDiease/', views.patientMyDiease),
    path('myDieaseHistory/', views.myDieaseHistory),
    path('myPatientInfo/', views.myPatientInfo),
    path('patientDieaseHistory/', views.patientDieaseHistory),
    path('patientMocaHistory/', views.patientMocaHistory),
    path('mocaDiagnosis/', views.mocaDiagnosis)
]