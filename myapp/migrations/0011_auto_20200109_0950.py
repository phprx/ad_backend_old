# Generated by Django 3.0 on 2020-01-09 01:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0010_auto_20200109_0932'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='moca_test',
            name='contest',
        ),
        migrations.AddField(
            model_name='moca_test',
            name='date',
            field=models.CharField(default='', max_length=50),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q1_1',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q1_2',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q1_3',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q2_1',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q2_2',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q2_3',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q3_1',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q3_2',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q4_1',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q4_2',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q4_3',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q4_4',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q5_1',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q5_2',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q5_3',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q6_1',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='moca_test',
            name='q7_1',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='moca_test',
            name='result',
            field=models.TextField(null=True),
        ),
    ]
