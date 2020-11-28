from django.db import models
import uuid


# Create your models here.


class doctor_info(models.Model):
    openId = models.CharField(primary_key=True, max_length=50, unique=True)
    name = models.CharField(max_length=50)
    gender = models.CharField(max_length=10)


class patient_info(models.Model):
    openId = models.CharField(primary_key=True, max_length=50, unique=True)
    name = models.CharField(max_length=50)
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    birthday = models.CharField(max_length=50)
    paramedic = models.OneToOneField("paramedic_info", on_delete=models.CASCADE, null=True)


class paramedic_info(models.Model):
    openId = models.CharField(primary_key=True, max_length=50, unique=True)
    name = models.CharField(max_length=50)
    gender = models.CharField(max_length=10)


class moca_test(models.Model):
    mocaTestId = models.AutoField(primary_key=True)
    q1_1 = models.TextField(null=True)
    q1_2 = models.TextField(null=True)
    q1_3 = models.TextField(null=True)
    q2_1 = models.TextField(null=True)
    q2_2 = models.TextField(null=True)
    q2_3 = models.TextField(null=True)
    q3_1 = models.TextField(null=True)
    q3_2 = models.TextField(null=True)
    q3_2_checkbox = models.TextField(null=True)
    q4_1 = models.TextField(null=True)
    q4_2 = models.TextField(null=True)
    q4_3 = models.TextField(null=True)
    q4_4 = models.TextField(null=True)
    q5_1 = models.TextField(null=True)
    q5_2 = models.TextField(null=True)
    q5_3 = models.TextField(null=True)
    q6_1 = models.TextField(null=True)
    q7_1 = models.TextField(null=True)
    status = models.CharField(max_length=50)
    result = models.TextField(null=True)
    date = models.CharField(max_length=50)
    patient = models.ForeignKey("patient_info", on_delete=models.CASCADE, null=True)
    doctor = models.ForeignKey("doctor_info", on_delete=models.CASCADE, null=True)


class patient_diease(models.Model):
    patientDieaseId = models.AutoField(primary_key=True)
    symptom = models.TextField(null=True)
    others = models.TextField(null=True)
    date = models.CharField(max_length=50)
    patient = models.ForeignKey("patient_info", on_delete=models.CASCADE, null=True)
