import json

from django.test import TestCase
import django
import os
import time

# Create your tests here.
django.setup()
from myapp import models

# models.doctor_info.objects.create(openId='2j2ij2j', name='潘豪', gender='男')        # 增加并返回创建的对象
# models.patient_info.objects.create(openId='382u8u3f3', name='小张', gender='男')
# doctor1 = models.doctor_info.objects.get(openId='o2AFI4xKgKvVv0QcgwVHfFa3RR5g')                            # 查询，没有会报错，不推荐用
# patient1 = models.patient_info.objects.get(openId='382u8u3f3')
# diagnosis1 = models.diagnosis_info.objects.create(type='moca', contest='进行了诊断', result='AD')
# models.diagnosis_info.objects.filter(diagnosisId=2).update(patient=patient1, doctor=doctor1)      # 修改
# models.diagnosis_info.objects.filter(diagnosisId=1).delete()                         # 删除
# models.diagnosis_info.objects.create(type='moca', contest='又进行了诊断', result='AD')
# result1 = models.diagnosis_info.objects.filter(type='moca')                            # 查询，获取的是一个列表，没查到不会报错
# if len(result1) != 0:
#     print(result1[0])
# else:
#     print('未获取到')

# if models.diagnosis_info.objects.filter(type='moca').exists():                      # 查询是否存在符合条件的记录
#     print('找到了')
# else:
#     print('未找到')

# tr = round(time.time())
# print('当前目录为：', tr)       #打印目录
# models.moca_test.objects.create(status='待诊断', date=time.strftime("%Y-%m-%d"), doctor=doctor1)        # 增加
# request.session.delete(sessionKey)