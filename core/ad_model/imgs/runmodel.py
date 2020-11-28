# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import numpy as np

model_path = "core/ad_model/imgs/clinical_m.pkl"
CN = np.array([[1, 0, 0]])
MCI = np.array([[0, 1, 0]])
AD = np.array([[0, 0, 1]])


def model_predict(image_address, clinical_data):
    clf = joblib.load(model_path)  # 模型地址
    y = clf.predict(clinical_data)  # [[0,0,0,0,0,0,0,0,0,0,0]]传入指标
    result = ""
    if (y == CN).all():
        result = "该患者目前正常，建议继续观察"
    if (y == MCI).all():
        result = "该患者属于认知障碍，建议到医院进一步诊断"
    if (y == AD).all():
        result = "该患者目前正常，建议继续观察"
    return result


if __name__ == "__main__":
    demo_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    demo_path = "/x/x/x/x/x"
    print(model_predict(demo_path, demo_data))
