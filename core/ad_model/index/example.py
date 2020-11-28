import numpy as np
# import pandas as pd

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.models import *


def diagnosis_AD_index(test):
    model = load_model('core/ad_model/index/adatten.m')
    # 模型输入：输入三次患者量表数据，顺序为['AGE','PTGENDER','FDG','MMSE','RAVLT.forgetting','FAQ','CDRSB.bl']

    # test = [[
    #     [77.3, 1, 5.575461927, 27.0, 8.0, 7.0, 1.5],
    #     [77.3, 1, 5.51286926, 25.0, 3.0, 2.0, 1.5],
    #     [77.3, 1, 5.317821432, 22.0, 2.0, 10.0, 1.5]
    # ]]
    test = np.array(test)
    print('test', test)
    test = test / 255.
    result = model.predict(test)
    for i in result:
        i = i.tolist()
    maxi = i.index(max(i))
    if maxi == 2:
        return '该患者有痴呆风险，建议到医院进行治疗'
    elif maxi == 1:
        return '该患者属于认知障碍，建议到医院进一步诊断'
    elif maxi == 0:
        return '该患者目前正常，建议继续观察'
    print(result)


if __name__ == "__main__":
    # test = [[
    #     [77.3, 1, 5.575461927, 27.0, 8.0, 7.0, 1.5],
    #     [77.3, 1, 5.51286926, 25.0, 3.0, 2.0, 1.5],
    #     [77.3, 1, 5.317821432, 22.0, 2.0, 10.0, 1.5]
    # ]]
    test = []
    input_data = [[], [], []]
    input_data[0].append(77.3)
    input_data[0].append(1)
    input_data[0].append(5.575461927)
    input_data[0].append(27.0)
    input_data[0].append(8.0)
    input_data[0].append(7.0)
    input_data[0].append(1.5)
    input_data[1].append(77.3)
    input_data[1].append(1)
    input_data[1].append(5.51286926)
    input_data[1].append(25.0)
    input_data[1].append(3.0)
    input_data[1].append(2.0)
    input_data[1].append(1.5)
    input_data[2].append(77.3)
    input_data[2].append(1)
    input_data[2].append(5.317821432)
    input_data[2].append(22.0)
    input_data[2].append(2.0)
    input_data[2].append(10.0)
    input_data[2].append(1.5)
    test.append(input_data)
    print(diagnosis_AD_index(test))
