# Generated by Django 3.0 on 2019-12-23 05:13

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0002_auto_20191223_1051'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='patient_info',
            name='doctor',
        ),
        migrations.RemoveField(
            model_name='patient_info',
            name='paramedic',
        ),
        migrations.CreateModel(
            name='diagnosis_info',
            fields=[
                ('diagnosisId', models.AutoField(primary_key=True, serialize=False)),
                ('type', models.CharField(max_length=50)),
                ('contest', models.TextField()),
                ('result', models.TextField()),
                ('doctor', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='myapp.doctor_info')),
                ('patient', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='myapp.patient_info')),
            ],
        ),
    ]
