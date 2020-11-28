import os
import time

from django.shortcuts import render
from django.http import HttpResponse
from . import models
import requests
import json
import datetime


# Create your views here.


def calculate(request):  # 这个接口是我这边用来测试的，AD这边用不到
    formula = request.GET['formula']
    request.session['test'] = 'session正常使用'  # 测试sessionid是否正常使用
    try:
        result = eval(formula, {})
    except:
        result = 'Error formula'
    return HttpResponse(result)


def getopenid(request):
    resp = None
    # print(request.session['test'])      # 测试sessionid是否正常使用
    # request.session['test'] = 'session正常使用'  # 测试sessionid是否正常使用
    if request.method == 'GET':
        payload = {'appid': 'wx6c40eb3a67e35f23', 'secret': '3b1f1ccd659ab577e24cd9ead98ac07d',
                   'js_code': request.GET['code'],
                   'grant_type': 'authorization_code'}
        resp = requests.post("https://api.weixin.qq.com/sns/jscode2session", params=payload)
        # print('请求的url：' + resp.url)
        # print('请求结果：' + resp.text)
        # print('openid：' + resp.json().get('openid'))
    return HttpResponse(json.dumps(resp.json()), content_type="application/json")


def login(request):
    resp = {}
    if request.method == 'GET':
        openId = request.GET.get('openId')
        request.session['openId'] = openId  # 登录时将用户的openId存入session
        print(openId)
        isDoctor = models.doctor_info.objects.filter(openId=openId).exists()  # 从医生表中查询是否存在此人
        isPatient = models.patient_info.objects.filter(openId=openId).exists()  # 从患者表中查询是否存在此人
        isParamedic = models.paramedic_info.objects.filter(openId=openId).exists()  # 从护理人员表中查询是否存在此人
        if isDoctor or isPatient or isParamedic:
            exist = True
        else:
            exist = False
        resp = {'isDoctor': isDoctor, 'isPatient': isPatient, 'isParamedic': isParamedic, 'exist': exist}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def modify(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        role = request.GET.get('role')
        name = request.GET.get('name')
        gender = request.GET.get('gender')
        age = request.GET.get('age')
        birthday = request.GET.get('timeChooser')
        if role == 'patient':
            models.patient_info.objects.create(openId=openId, name=name, gender=gender, age=age, birthday=birthday)
        if role == 'doctor':
            models.doctor_info.objects.create(openId=openId, name=name, gender=gender)
        if role == 'paramedic':
            models.paramedic_info.objects.create(openId=openId, name=name, gender=gender)
        resp = {'result': 'success'}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def AI_report(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        text = request.GET.get('text')
        from core.ad_model.text.predict_new import diagnosis_AD_text
        result = diagnosis_AD_text(text)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def AI_image(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        age = 60  # 默认60岁
        gender = int(1)  # 默认为男
        if models.patient_info.objects.filter(openId=openId).exists():
            age = int(models.patient_info.objects.get(openId=openId).age)
            origin_gender = models.patient_info.objects.get(openId=openId).gender
            if origin_gender == '女':
                gender = int(0)
        Abeta40 = float(request.GET.get('Abeta(40)'))
        Abeta42 = float(request.GET.get('Abeta(42)'))
        TAU = float(request.GET.get('TAU'))
        PTAU = float(request.GET.get('PTAU'))
        APOE = float(request.GET.get('APOE'))
        MMSE = float(request.GET.get('MMSE'))
        CDR = float(request.GET.get('CDR'))
        RAVLT = float(request.GET.get('RAVLT'))
        FAQ = float(request.GET.get('FAQ'))
        input_data = [[]]
        input_data[0].append(TAU)
        input_data[0].append(PTAU)
        input_data[0].append(Abeta40)
        input_data[0].append(Abeta42)
        input_data[0].append(age)
        input_data[0].append(gender)
        input_data[0].append(APOE)
        input_data[0].append(CDR)
        input_data[0].append(MMSE)
        input_data[0].append(RAVLT)
        input_data[0].append(FAQ)
        print('input', input_data)
        from core.ad_model.imgs.runmodel import model_predict
        result = model_predict('', input_data)
        resp = {'result': result}
        time.sleep(3)
    return HttpResponse(json.dumps(resp), content_type="application/json")


def AI_scale1(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        age = 60  # 默认60岁
        gender = int(0)  # 默认为男
        if models.patient_info.objects.filter(openId=openId).exists():
            age = int(models.patient_info.objects.get(openId=openId).age)
            origin_gender = models.patient_info.objects.get(openId=openId).gender
            if origin_gender == '女':
                gender = int(1)
        test = []
        input_data = [[], [], []]
        input_data[0].append(age)
        input_data[0].append(gender)
        input_data[0].append(float(request.GET.get('first_FDG')))
        input_data[0].append(float(request.GET.get('first_MMSE')))
        input_data[0].append(float(request.GET.get('first_RAVLT')))
        input_data[0].append(float(request.GET.get('first_FAQ')))
        input_data[0].append(float(request.GET.get('first_CDRSB')))
        input_data[1].append(age)
        input_data[1].append(gender)
        input_data[1].append(float(request.GET.get('second_FDG')))
        input_data[1].append(float(request.GET.get('second_MMSE')))
        input_data[1].append(float(request.GET.get('second_RAVLT')))
        input_data[1].append(float(request.GET.get('second_FAQ')))
        input_data[1].append(float(request.GET.get('second_CDRSB')))
        input_data[2].append(age)
        input_data[2].append(gender)
        input_data[2].append(float(request.GET.get('third_FDG')))
        input_data[2].append(float(request.GET.get('third_MMSE')))
        input_data[2].append(float(request.GET.get('third_RAVLT')))
        input_data[2].append(float(request.GET.get('third_FAQ')))
        input_data[2].append(float(request.GET.get('third_CDRSB')))
        test.append(input_data)
        from core.ad_model.index.example import diagnosis_AD_index
        result = diagnosis_AD_index(test)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def AI_scale2(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        age = '60'  # 默认60岁
        gender = 'Male'  # 默认为男
        if models.patient_info.objects.filter(openId=openId).exists():
            age = str(models.patient_info.objects.get(openId=openId).age)
            origin_gender = models.patient_info.objects.get(openId=openId).gender
            if origin_gender == '女':
                gender = 'Female'
        data = {"PTID": ['1', '2'], "Age": [age, age], "Sex": [gender, gender],
                "ADAS13": [request.GET.get('first_ADAS13'), request.GET.get('second_ADAS13')],
                "MMSE": [request.GET.get('first_MMSE'), request.GET.get('second_MMSE')],
                "Hippocampus": [request.GET.get('first_Hippocampus'), request.GET.get('second_Hippocampus')],
                "WholeBrain": [request.GET.get('first_Whole Brain'), request.GET.get('second_Whole Brain')],
                "ICV": [request.GET.get('first_ICV'), request.GET.get('second_ICV')],
                "EXAMDATE": [request.GET.get('first_Timechooser'), request.GET.get('second_Timechooser')],
                "ABETA": [request.GET.get('first_Abeta'), request.GET.get('second_Abeta')],
                "TAU": [request.GET.get('first_TAU'), request.GET.get('second_TAU')],
                "PTAU": [request.GET.get('first_PTAU'), request.GET.get('second_PTAU')]}
        import pandas as pd
        DataTest = pd.DataFrame(data)
        print(DataTest)
        from core.ad_model.index2.predict import predict_AD_index
        result = predict_AD_index(DataTest)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def report_content(request):
    if request.method == 'GET':
        return HttpResponse("服务器不接受GET请求!")
    else:
        openId = request.session['openId']
        print(openId)
        report_file = request.FILES.get('file')
        # file_name = image_file.name
        # file_size = image_file.size
        # os.mkdir(os.getcwd() + '/report')
        timestamp = str(round(time.time()))
        path = 'report/' + openId + '_' + timestamp + '.txt'
        fw = open(path, 'wb+')
        for chunk in report_file.chunks():
            fw.write(chunk)
        fw.close()
        fr = open(path, 'r', encoding='GBK')
        content = fr.read()
        fr.close()
        return HttpResponse(content)


# def imageTest(request):
#     if request.method == 'GET':
#         return HttpResponse("服务器不接受GET请求!")
#     else:
#         openId = request.session['openId']
#         print(openId)
#         if not os.path.exists('moca/' + openId + '/'):
#             os.makedirs('moca/' + openId + '/')
#         report_file = request.FILES.get('file')
#         timestamp = str(round(time.time()))
#         path = 'moca/' + openId + '_' + timestamp + '.png'
#         fw = open(path, 'wb+')
#         for chunk in report_file.chunks():
#             fw.write(chunk)
#         fw.close()
#         return HttpResponse()
#
#
# def audioTest(request):
#     if request.method == 'GET':
#         return HttpResponse("服务器不接受GET请求!")
#     else:
#         openId = request.session['openId']
#         print(openId)
#         print(request.POST.get('user'))
#         if not os.path.exists('moca/' + openId + '/'):
#             os.makedirs('moca/' + openId + '/')
#         report_file = request.FILES.get('3-1')
#         timestamp = str(round(time.time()))
#         name = '3-1_' + timestamp + '.mp3'
#         path = 'moca/' + openId + '/'
#         fw = open(path + name, 'wb+')
#         for chunk in report_file.chunks():
#             fw.write(chunk)
#         fw.close()
#         return HttpResponse()


def mutifile(request):
    if request.method == 'GET':
        openId = request.session['openId']
        q1_1 = request.session['1-1']
        q1_2 = request.session['1-2']
        q1_3 = request.session['1-3']
        q2_1 = request.GET.get('2-1')
        q2_2 = request.GET.get('2-2')
        q2_3 = request.GET.get('2-3')
        q3_1 = request.session['3-1']
        q3_2 = request.session['3-2']
        q3_2_checkbox = request.GET.get('3-2checkbox')
        q4_1 = request.session['4-1']
        q4_2 = request.session['4-2']
        q4_3 = request.GET.get('4-3')
        q4_4 = request.GET.get('4-4')
        q5_1 = request.session['5-1']
        q5_2 = request.session['5-2']
        q5_3 = request.session['5-3']
        q6_1 = request.GET.get('6-1')
        q7_1 = request.GET.get('7-1')

        patient = models.patient_info.objects.get(openId=openId)
        models.moca_test.objects.create(status='待诊断', date=time.strftime("%Y-%m-%d"), patient=patient, q1_1=q1_1,
                                        q1_2=q1_2, q1_3=q1_3, q2_1=q2_1, q2_2=q2_2, q2_3=q2_3, q3_1=q3_1, q3_2=q3_2,
                                        q3_2_checkbox=q3_2_checkbox, q4_1=q4_1, q4_2=q4_2, q4_3=q4_3, q4_4=q4_4,
                                        q5_1=q5_1, q5_2=q5_2, q5_3=q5_3, q6_1=q6_1, q7_1=q7_1)
        # 清除moca相关的session
        del request.session['1-1']
        del request.session['1-2']
        del request.session['1-3']
        del request.session['3-1']
        del request.session['3-2']
        del request.session['4-1']
        del request.session['4-2']
        del request.session['5-1']
        del request.session['5-2']
        del request.session['5-3']
        return HttpResponse()
    else:
        openId = request.session['openId']
        print(openId)
        n = str(request.POST.get('NUMBER'))
        print('第' + n + '个文件上传成功')
        if not os.path.exists('moca/' + openId + '/'):
            os.makedirs('moca/' + openId + '/')
        report_file = request.FILES.get('file')
        timestamp = str(round(time.time()))
        fileType = 'png'
        if int(n) > 2:
            fileType = 'mp3'

        if int(n) == 0:
            index = '1-1'
        elif int(n) == 1:
            index = '1-2'
        elif int(n) == 2:
            index = '1-3'
        elif int(n) == 3:
            index = '3-1'
        elif int(n) == 4:
            index = '3-2'
        elif int(n) == 5:
            index = '4-1'
        elif int(n) == 6:
            index = '4-2'
        elif int(n) == 7:
            index = '5-1'
        elif int(n) == 8:
            index = '5-2'
        else:
            index = '5-3'
        name = index + '_' + timestamp + '.' + fileType
        path = 'moca/' + openId + '/'
        fw = open(path + name, 'wb+')
        for chunk in report_file.chunks():
            fw.write(chunk)
        fw.close()
        request.session[index] = path + name
        return HttpResponse()


def myMocaHistory(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        result = []
        patient = models.patient_info.objects.get(openId=openId)
        mocaList = models.moca_test.objects.filter(patient=patient)
        if len(mocaList)!=0:
            for moca in mocaList:
                if moca.doctor is None:
                    doctorName = '--'
                else:
                    doctorName = moca.doctor.name
                if moca.result is None:
                    mocaResult = '--'
                else:
                    mocaResult = moca.result
                mocaDict = {'mocaTestId': moca.mocaTestId, 'date': moca.date, 'doctorName': doctorName,
                            'status': moca.status, 'result': mocaResult}
                result.append(mocaDict)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def patientMyDiease(request):
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        symptom = request.GET.get('checkbox')
        others = request.GET.get('textarea')
        date = request.GET.get('timeChooser')
        patient = models.patient_info.objects.get(openId=openId)
        models.patient_diease.objects.create(symptom=symptom, others=others, date=date, patient=patient)
    return HttpResponse()


def myDieaseHistory(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        patient = models.patient_info.objects.get(openId=openId)
        histories = models.patient_diease.objects.filter(patient=patient)
        result = []
        if len(histories) != 0:
            for history in histories:
                historyDict = {'time': history.date, 'text': history.symptom, 'type1': history.others}
                result.append(historyDict)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def myPatientInfo(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        doctor = models.doctor_info.objects.get(openId=openId)
        mocas = models.moca_test.objects.filter(doctor=doctor)
        originResult = []
        result = []
        if len(mocas) != 0:
            for moca in mocas:
                infoDict = {'id': moca.patient.openId, 'name': moca.patient.name, 'gender': moca.patient.gender,
                            'age': moca.patient.age}
                originResult.append(infoDict)
            for r in originResult:  # 去除重复元素
                if r not in result:
                    result.append(r)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def patientDieaseHistory(request):
    resp = {}
    if request.method == 'GET':
        openId = request.GET.get('patientOpenid')
        print(openId)
        patient = models.patient_info.objects.get(openId=openId)
        histories = models.patient_diease.objects.filter(patient=patient)
        result = []
        if len(histories) != 0:
            for history in histories:
                historyDict = {'time': history.date, 'text': history.symptom, 'type1': history.others}
                result.append(historyDict)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def patientMocaHistory(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        result = []
        doctor = models.doctor_info.objects.get(openId=openId)
        mocaList = models.moca_test.objects.filter(doctor=doctor)
        if len(mocaList) != 0:
            for moca in mocaList:
                patientName = moca.patient.name
                if moca.result is None:
                    mocaResult = '--'
                else:
                    mocaResult = moca.result
                mocaDict = {'mocaTestId': moca.mocaTestId, 'date': moca.date, 'patientName': patientName,
                            'status': moca.status, 'result': mocaResult}
                result.append(mocaDict)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def mocaDiagnosis(request):
    resp = {}
    if request.method == 'GET':
        openId = request.session['openId']
        print(openId)
        result = []
        mocaList = models.moca_test.objects.filter(status='待诊断')
        if len(mocaList) != 0:
            for moca in mocaList:
                patientName = moca.patient.name
                patientGender = moca.patient.gender
                patientAge = moca.patient.age
                mocaDict = {'id': moca.mocaTestId, 'date': moca.date, 'name': patientName, 'gender': patientGender,
                            'age': patientAge}
                result.append(mocaDict)
        resp = {'result': result}
    return HttpResponse(json.dumps(resp), content_type="application/json")