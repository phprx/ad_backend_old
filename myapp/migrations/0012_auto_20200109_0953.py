# Generated by Django 3.0 on 2020-01-09 01:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0011_auto_20200109_0950'),
    ]

    operations = [
        migrations.AlterField(
            model_name='moca_test',
            name='date',
            field=models.CharField(max_length=50),
        ),
    ]